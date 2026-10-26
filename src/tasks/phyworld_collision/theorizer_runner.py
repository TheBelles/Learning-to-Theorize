"""Train NEO on PhyWorld collision episodes.

Single GPU::

    python -m tasks.phyworld_collision.theorizer_runner \\
        --train-h5 data/collision/train.h5 --test-h5 data/collision/id_test.h5 \\
        --observation-checkpoint runs/collision/observation/pixels-*/checkpoints/best.pt \\
        --output-root runs/collision/neo --seed 42

Several GPUs (data parallel; the batch size is per GPU)::

    torchrun --standalone --nproc-per-node=2 -m tasks.phyworld_collision.theorizer_runner ...

Change any setting of :class:`experiment_config.TrainingConfig` with
``--set name=value`` (repeatable), e.g. ``--set action_codebook_size=4
--set epochs=100``.  The observation model is frozen; the run directory gets
``checkpoints/best.pt`` (lowest ID query transfer loss, the paper's selection
rule), periodic checkpoints and ``final.pt``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, Subset

from tasks.phyworld_collision.data.dataset import (
    EpisodeBatch,
    EpisodeDataset,
    collate_episodes,
    load_metadata,
)
from tasks.phyworld_collision.experiment_config import Experiment, TrainingConfig, apply_overrides
from tasks.phyworld_collision.task import (
    build_neo,
    load_observation_checkpoint,
    load_observation_weights,
)
from training.optimization import (
    OptimizationConfig,
    build_cosine_warmup_scheduler,
    optimization_step,
)
from training.runtime import (
    ExperimentTrackingConfig,
    MetricLog,
    barrier,
    bfloat16_autocast,
    capture_git_source,
    create_run_directory,
    initialize_distributed,
    seed_everything,
)


def build_optimizer(model: torch.nn.Module, training: TrainingConfig) -> AdamW:
    """AdamW with the paper's slower programmer and executor learning rates."""

    programmer, executor, remainder = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("theory_programmer."):
            programmer.append(parameter)
        elif name.startswith("program_executor."):
            executor.append(parameter)
        else:
            remainder.append(parameter)
    base = training.learning_rate
    groups = [
        {"params": programmer, "lr": base * training.policy_learning_rate_scale},
        {"params": executor, "lr": base * training.transition_learning_rate_scale},
        {"params": remainder, "lr": base},
    ]
    return AdamW([g for g in groups if g["params"]], lr=base, weight_decay=training.weight_decay)


def forward(model, batch: EpisodeBatch, *, is_eval: bool, coefficient: float, num_transitions=None):
    with bfloat16_autocast(batch.observations.device):
        return model(
            batch.observations,
            is_eval=is_eval,
            length_control_coefficient=coefficient,
            num_transitions=num_transitions,
        )


@torch.no_grad()
def evaluate(model, loader, device, *, coefficient: float, deltas: np.ndarray) -> dict[str, float]:
    """ID query transfer: program inferred on the support pair, applied to the query pair."""

    model.eval()
    keys = (
        "support_loss",
        "query_loss",
        "query_l1",
        "query_foreground_l1",
        "mean_explanation_length",
        "length_matches_delta",
    )
    sums = dict.fromkeys(keys, 0.0)
    count = 0
    codes_used: set[int] = set()
    for batch in loader:
        batch = batch.to(device)
        output = forward(model, batch, is_eval=True, coefficient=coefficient)
        support_lengths = output.selected_lengths[::2].cpu().numpy()
        true_deltas = deltas[batch.indices.numpy()]
        record = {
            "support_loss": float(output.reconstruction_loss),
            "query_loss": float(output.query_reconstruction_loss),
            "query_l1": float(output.query_metrics.l1),
            "query_foreground_l1": float(output.query_metrics.foreground_l1),
            "mean_explanation_length": float(support_lengths.mean()),
            "length_matches_delta": float((support_lengths == true_deltas).mean()),
        }
        for key, value in record.items():
            sums[key] += value * len(batch)
        count += len(batch)
        codes_used.update(output.action_vq_by_length[0].indices[::2].cpu().tolist())
    if torch.distributed.is_initialized():
        totals = torch.tensor([*sums.values(), count], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(totals)
        sums = dict(zip(sums, totals[:-1].tolist()))
        count = totals[-1].item()
    result = {key: value / max(count, 1) for key, value in sums.items()}
    result["codes_used_first_step"] = float(len(codes_used))
    return result


def run(arguments: argparse.Namespace) -> Path:
    context = initialize_distributed(force_cpu=arguments.cpu)
    seed_everything(arguments.seed + context.rank)
    device = context.device

    observation, observation_state = load_observation_checkpoint(arguments.observation_checkpoint)
    training = apply_overrides(TrainingConfig(), arguments.set or [])
    training = replace(
        training, state_dim=observation.state_dim, num_state_tokens=observation.num_state_tokens
    )
    experiment = Experiment(training)

    train_set = EpisodeDataset(arguments.train_h5, observation.modality)
    test_set = EpisodeDataset(arguments.test_h5, observation.modality)
    data_deltas = [int(d) for d in np.atleast_1d(train_set.attributes["deltas"])]
    if max(data_deltas) > training.max_transition_length:
        print(
            f"warning: training deltas go up to {max(data_deltas)} but max_transition_length is "
            f"{training.max_transition_length}; a program cannot apply one primitive per step",
            flush=True,
        )
    if observation.modality == "pixels" and int(train_set.attributes["image_size"]) != observation.image_size:
        raise ValueError("episode image_size differs from the observation model's")
    test_deltas = load_metadata(test_set.path).delta

    model = build_neo(experiment, observation)
    loaded = load_observation_weights(model, observation_state)
    model.to(device)
    core = model
    if context.is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[context.local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,  # the programmer keeps unused FiLM parameters
        )

    sampler = None
    if context.is_distributed:
        sampler = DistributedSampler(
            train_set, num_replicas=context.world_size, rank=context.rank, shuffle=True, seed=arguments.seed
        )
        test_subset = Subset(test_set, range(context.rank, len(test_set), context.world_size))
    else:
        test_subset = test_set
    generator = torch.Generator().manual_seed(arguments.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=training.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=training.num_workers,
        collate_fn=collate_episodes,
        pin_memory=device.type == "cuda",
        drop_last=len(train_set) > training.batch_size * context.world_size,
        generator=generator,
        persistent_workers=training.num_workers > 0,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        collate_fn=collate_episodes,
    )

    total_steps = training.epochs * max(len(train_loader), 1)
    optimization = OptimizationConfig(
        total_steps=total_steps,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        max_gradient_norm=training.gradient_clip_norm,
        warmup_ratio=training.warmup_ratio,
        minimum_learning_rate_ratio=training.minimum_learning_rate_ratio,
    )
    optimizer = build_optimizer(core, training)
    scheduler = build_cosine_warmup_scheduler(optimizer, optimization)

    run_name = arguments.run_name or f"collision-neo-{observation.modality}-seed{arguments.seed}"
    run_directory = create_run_directory(
        arguments.output_root,
        run_name,
        capture_git_source(Path(__file__).resolve().parents[3]),
        context,
    )
    resolved = {
        "task": "phyworld_collision",
        "seed": arguments.seed,
        "train_h5": str(train_set.path),
        "test_h5": str(test_set.path),
        "observation_checkpoint": str(Path(arguments.observation_checkpoint).resolve()),
        "observation_tensors_loaded": loaded,
        "total_steps": total_steps,
        "world_size": context.world_size,
        "data_deltas": data_deltas,
        "frame_step": int(train_set.attributes["frame_step"]),
        **{f"observation.{k}": v for k, v in observation.to_dict().items()},
        **{f"training.{k}": v for k, v in training.to_dict().items()},
    }
    log: MetricLog | None = None
    if context.is_global_zero:
        (run_directory / "checkpoints").mkdir(exist_ok=True)
        (run_directory / "config.json").write_text(json.dumps(resolved, indent=2))
        log = MetricLog(
            run_directory / "metrics.jsonl",
            ExperimentTrackingConfig(
                project=arguments.wandb_project,
                name=run_name,
                entity=arguments.wandb_entity,
                group=arguments.wandb_group,
                mode=arguments.wandb_mode,
            ),
            resolved,
        )

    def coefficient_at(step: int) -> float:
        return experiment.scheduled_length_control(step / max(total_steps, 1))

    def save(name: str, epoch: int, global_step: int, evaluation: dict) -> None:
        torch.save(
            {
                "model_state_dict": core.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "training_config": training.to_dict(),
                "observation_config": observation.to_dict(),
                "frame_step": int(train_set.attributes["frame_step"]),
                "epoch": epoch,
                "global_step": global_step,
                "evaluation": evaluation,
                "config": resolved,
            },
            run_directory / "checkpoints" / name,
        )

    exit_code = 1
    best = math.inf
    evaluation: dict[str, float] = {}
    try:
        global_step = 0
        started = time.monotonic()
        for epoch in range(training.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in train_loader:
                coefficient = coefficient_at(global_step)
                result = optimization_step(
                    model,
                    batch.to(device),
                    lambda m, b, c=coefficient: forward(m, b, is_eval=False, coefficient=c),
                    optimizer,
                    scheduler,
                    global_step=global_step,
                    max_gradient_norm=training.gradient_clip_norm,
                )
                core.quantizer.step()
                global_step = result.global_step
                if log is not None and (
                    global_step % training.log_interval_steps == 0 or global_step == total_steps
                ):
                    output = result.objective
                    log.log(
                        {
                            "train/step": global_step,
                            "train/epoch": epoch,
                            "train/loss": float(output.loss.detach()),
                            "train/reconstruction_loss": float(output.reconstruction_loss.detach()),
                            "train/grounding_loss": float(output.transition_consistency_loss.detach()),
                            "train/state_prediction_loss": float(output.state_prediction_loss.detach()),
                            "train/action_vq_loss": float(output.action_vq_loss.detach()),
                            "train/foreground_l1": float(output.metrics.foreground_l1),
                            "train/mean_explanation_length": float(output.mean_explanation_length),
                            "train/gradient_norm": float(result.gradient_norm),
                            "train/learning_rate": result.learning_rates[-1],
                            "train/length_control_coefficient": coefficient,
                        }
                    )
            evaluation = evaluate(
                core, test_loader, device, coefficient=coefficient_at(global_step), deltas=test_deltas
            )
            if log is not None:
                log.log({"eval/step": global_step, "eval/epoch": epoch, **{f"eval/{k}": v for k, v in evaluation.items()}})
                print(json.dumps({"epoch": epoch, "step": global_step, **evaluation}), flush=True)
            if context.is_global_zero:
                if evaluation["query_loss"] < best:
                    best = evaluation["query_loss"]
                    save("best.pt", epoch, global_step, evaluation)
                if (epoch + 1) % training.checkpoint_interval_epochs == 0:
                    save(f"checkpoint_{global_step}.pt", epoch, global_step, evaluation)
        if context.is_global_zero:
            save("final.pt", training.epochs - 1, global_step, evaluation)
            (run_directory / "result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "global_step": global_step,
                        "seconds": time.monotonic() - started,
                        "best_query_loss": best,
                        "final_evaluation": evaluation,
                    },
                    indent=2,
                )
            )
        exit_code = 0
    finally:
        if log is not None:
            log.finish(exit_code=exit_code)
        barrier(context)
        if context.owns_process_group:
            torch.distributed.destroy_process_group()
    return run_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--test-h5", type=Path, required=True)
    parser.add_argument("--observation-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--set", action="append", metavar="NAME=VALUE", help="override a TrainingConfig field")
    parser.add_argument("--run-name")
    parser.add_argument("--wandb-project", default="LearningToTheorize")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--cpu", action="store_true", help="force CPU (smoke tests)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run_directory = run(arguments)
    print(json.dumps({"run_directory": str(run_directory)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
