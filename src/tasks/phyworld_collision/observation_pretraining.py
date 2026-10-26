"""Pretrain the observation VAE on single observations of the episode files.

NEO keeps this model frozen, so its reconstructions bound everything NEO can
predict.  Check ``reconstructions_epoch*.png`` in the run directory: both
balls must be sharp and in the right place before you train NEO.

Pixels (default)::

    python -m tasks.phyworld_collision.observation_pretraining \\
        --train-h5 data/collision/train.h5 --test-h5 data/collision/id_test.h5 \\
        --output-root runs/collision/observation --epochs 30

States::

    python -m tasks.phyworld_collision.observation_pretraining --modality states \\
        --state-dim 64 --train-h5 ... --test-h5 ... --output-root ... --epochs 50

The run directory gets ``checkpoints/best.pt`` (lowest test loss) and
``checkpoints/final.pt``; pass either to the NEO runner.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tasks.phyworld_collision.data.dataset import ObservationDataset
from tasks.phyworld_collision.models.vae import ObservationConfig, ObservationVAE
from tasks.phyworld_collision.objective import PixelObjective, StateObjective
from tasks.phyworld_collision.visualize import save_rows
from training.runtime import bfloat16_autocast, initialize_distributed, seed_everything


def _loss_function(config: ObservationConfig, foreground_weight: float):
    if config.modality == "pixels":
        objective = PixelObjective(config.image_size, config.channels, foreground_weight)
    else:
        objective = StateObjective(config.input_dim)
    return objective


def _evaluate(model, loader, objective, device) -> dict[str, float]:
    model.eval()
    total = {"loss": 0.0, "l1": 0.0, "foreground_l1": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with bfloat16_autocast(device):
                reconstruction, _ = model(batch, sample=False)
            loss = objective.reconstruction_loss(reconstruction.float(), batch)
            metrics = objective.metrics(reconstruction.float(), batch)
            n = batch.shape[0]
            total["loss"] += float(loss) * n
            total["l1"] += metrics.l1 * n
            total["foreground_l1"] += metrics.foreground_l1 * n
            count += n
    return {key: value / max(count, 1) for key, value in total.items()}


def _save_reconstructions(model, dataset, device, path: Path, count: int = 8) -> None:
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, count).astype(int)
    batch = torch.stack([dataset[int(i)] for i in indices]).to(device)
    with torch.no_grad():
        reconstruction, _ = model(batch, sample=False)
    rows = [[batch[i].cpu().numpy(), reconstruction[i].float().cpu().numpy()] for i in range(count)]
    save_rows(path, rows)


def run(arguments: argparse.Namespace) -> Path:
    seed_everything(arguments.seed)
    context = initialize_distributed(force_cpu=arguments.cpu)
    if context.is_distributed:
        raise SystemExit("observation pretraining runs on one device; launch without torchrun")
    device = context.device

    train_set = ObservationDataset(arguments.train_h5, arguments.modality)
    test_set = ObservationDataset(arguments.test_h5, arguments.modality)
    attributes = train_set.attributes
    config = ObservationConfig(
        modality=arguments.modality,
        state_dim=arguments.state_dim,
        dropout=arguments.dropout,
        image_size=int(attributes["image_size"]),
        base_channels=arguments.base_channels,
        bottleneck_channels=arguments.bottleneck_channels,
        bottleneck_size=arguments.bottleneck_size,
        hidden_dim=arguments.hidden_dim,
    )
    objective = _loss_function(config, arguments.foreground_weight)
    model = ObservationVAE(config).to(device)

    if arguments.max_test_observations and len(test_set) > arguments.max_test_observations:
        picked = np.linspace(0, len(test_set) - 1, arguments.max_test_observations).astype(int)
        test_subset = Subset(test_set, picked.tolist())
    else:
        test_subset = test_set
    generator = torch.Generator().manual_seed(arguments.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_set) > arguments.batch_size,
        generator=generator,
        persistent_workers=arguments.num_workers > 0,
    )
    test_loader = DataLoader(
        test_subset, batch_size=arguments.batch_size, shuffle=False, num_workers=arguments.num_workers
    )

    total_steps = arguments.epochs * max(len(train_loader), 1)
    warmup = int(0.02 * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=0.0)

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)

    stamp = time.strftime("%Y%m%dT%H%M%S")
    run_directory = Path(arguments.output_root).expanduser().resolve() / f"{config.modality}-{stamp}"
    (run_directory / "checkpoints").mkdir(parents=True, exist_ok=False)
    resolved = {
        "observation_config": config.to_dict(),
        "train_h5": str(train_set.path),
        "test_h5": str(test_set.path),
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(arguments).items()},
    }
    (run_directory / "config.json").write_text(json.dumps(resolved, indent=2))
    log = (run_directory / "metrics.jsonl").open("x")

    def save(name: str, epoch: int, metrics: dict) -> Path:
        path = run_directory / "checkpoints" / name
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "observation_config": config.to_dict(),
                "epoch": epoch,
                "metrics": metrics,
            },
            path,
        )
        return path

    best = math.inf
    step = 0
    started = time.monotonic()
    try:
        for epoch in range(arguments.epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(device, non_blocking=True)
                with bfloat16_autocast(device):
                    reconstruction, posterior = model(batch)
                reconstruction_loss = objective.reconstruction_loss(reconstruction.float(), batch)
                kl = posterior.kl_to_standard_normal().float()
                loss = reconstruction_loss + arguments.kl_weight * kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                step += 1
                if step % arguments.log_interval == 0:
                    record = {
                        "step": step,
                        "epoch": epoch,
                        "train/loss": float(loss),
                        "train/reconstruction": float(reconstruction_loss),
                        "train/kl": float(kl),
                        "train/gradient_norm": float(gradient_norm),
                        "train/lr": scheduler.get_last_lr()[0],
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
            metrics = _evaluate(model, test_loader, objective, device)
            record = {"step": step, "epoch": epoch, **{f"test/{k}": v for k, v in metrics.items()}}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if config.modality == "pixels":
                _save_reconstructions(
                    model, test_set, device, run_directory / f"reconstructions_epoch{epoch:03d}.png"
                )
            if metrics["loss"] < best:
                best = metrics["loss"]
                save("best.pt", epoch, metrics)
        save("final.pt", arguments.epochs - 1, metrics)
    finally:
        log.close()
    (run_directory / "result.json").write_text(
        json.dumps(
            {"best_test_loss": best, "seconds": time.monotonic() - started, "final": metrics},
            indent=2,
        )
    )
    return run_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--test-h5", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--modality", choices=("pixels", "states"), default="pixels")
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--bottleneck-channels", type=int, default=256)
    parser.add_argument("--bottleneck-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256, help="MLP width (states)")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--kl-weight", type=float, default=1e-5)
    parser.add_argument("--foreground-weight", type=float, default=10.0)
    parser.add_argument("--max-test-observations", type=int, default=8000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="force CPU (smoke tests)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run_directory = run(arguments)
    print(json.dumps({"run_directory": str(run_directory)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
