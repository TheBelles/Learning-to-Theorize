"""Evaluate a trained NEO on any number of episode files.

::

    python -m tasks.phyworld_collision.evaluation \\
        --checkpoint runs/collision/neo/.../checkpoints/best.pt \\
        --split id=data/collision/id_test.h5 \\
        --split length_ood=data/collision/length_ood.h5 \\
        --split velocity_ood=data/collision/velocity_ood.h5 \\
        --output-dir runs/collision/eval

For each split the program is inferred on the **support** pair (including its
length, chosen by MDL) and executed on the **query** pair.  Reported:

* ``query_loss`` / ``query_foreground_l1``: image (or state) error of the
  transferred program, the L2T metric;
* ``length_matches_delta``: how often the selected program length equals the
  number of steps ``delta`` (did NEO learn "one primitive = one step"?);
* physics errors of the query prediction in world units: ball **position**
  at the last frame and ball **velocity**, measured by tracking the red and
  blue balls (pixels) or read off the state vector (states).  Balls that are
  not fully inside the target frame are excluded;
* two reference predictors on the same episodes: ``copy`` (nothing moves) and
  ``constant_velocity`` (true input positions extrapolated, no collision);
* everything split by whether the query window contains the contact and by
  ``delta``, plus which codebook entries the programs use.

The number of program steps defaults to ``max(max_transition_length, largest
delta in the split)`` so length-OOD splits can use longer programs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from tasks.phyworld_collision.data.dataset import EpisodeDataset, collate_episodes, load_metadata
from tasks.phyworld_collision.physics import (
    FRAME_DT,
    WORLD_SCALE,
    ball_centroids,
    state_to_physical,
)
from tasks.phyworld_collision.task import load_neo_checkpoint
from tasks.phyworld_collision.visualize import save_rows
from training.runtime import bfloat16_autocast


def _fully_inside(positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """``positions`` (N, 2, 2) and ``radii`` (N, 2) -> (N, 2) bool."""

    low = positions - radii[..., None]
    high = positions + radii[..., None]
    return np.all((low >= 0.0) & (high <= WORLD_SCALE), axis=-1)


def physics_from_pixels(observations: np.ndarray, frame_step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tracked last-frame centres, velocities and visibility from ``(N, S, S, 6)``."""

    earlier = ball_centroids(observations[..., :3])
    later = ball_centroids(observations[..., 3:6])
    velocity = (later.centers - earlier.centers) / (frame_step * FRAME_DT)
    return later.centers, velocity, earlier.visible & later.visible


def physics_from_states(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = state_to_physical(states)
    visible = np.ones(values["position"].shape[:2], dtype=bool)
    return values["position"], values["velocity"], visible


def _summary(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask]
    selected = selected[np.isfinite(selected)]
    return float(selected.mean()) if selected.size else None


@torch.no_grad()
def evaluate_split(
    model,
    path: Path,
    *,
    batch_size: int,
    num_transitions: int | None,
    device: torch.device,
    visualize: int,
    output_dir: Path,
    name: str,
) -> dict[str, Any]:
    observation = model.observation_config
    training = model.experiment.training
    dataset = EpisodeDataset(path, observation.modality)
    metadata = load_metadata(path)
    frame_step = int(dataset.attributes["frame_step"])
    transitions = num_transitions or max(training.max_transition_length, int(metadata.delta.max()))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_episodes)
    objective = model.objective
    coefficient = training.length_control_end

    per_episode: dict[str, list[np.ndarray]] = {
        k: [] for k in ("index", "support_loss", "query_loss", "length", "codes", "pred_pos", "pred_vel", "pred_visible")
    }
    examples: list[list[np.ndarray]] = []
    for batch in loader:
        batch = batch.to(device)
        with bfloat16_autocast(device):
            output = model(
                batch.observations,
                is_eval=True,
                length_control_coefficient=coefficient,
                num_transitions=transitions,
            )
        inputs, targets = objective.split_batch(batch.observations)
        predictions = output.predictions.float()
        per_episode["index"].append(batch.indices.numpy())
        per_episode["support_loss"].append(
            objective.per_sample_reconstruction_loss(predictions[::2], targets[::2]).float().cpu().numpy()
        )
        per_episode["query_loss"].append(
            objective.per_sample_reconstruction_loss(predictions[1::2], targets[1::2]).float().cpu().numpy()
        )
        per_episode["length"].append(output.selected_lengths[::2].cpu().numpy())
        per_episode["codes"].append(
            torch.stack([step.indices[::2] for step in output.action_vq_by_length], dim=1).cpu().numpy()
        )
        query_prediction = objective.decode_prediction(predictions[1::2]).float().cpu().numpy()
        if observation.modality == "pixels":
            position, velocity, visible = physics_from_pixels(query_prediction, frame_step)
        else:
            position, velocity, visible = physics_from_states(query_prediction)
        per_episode["pred_pos"].append(position)
        per_episode["pred_vel"].append(velocity)
        per_episode["pred_visible"].append(visible)
        if observation.modality == "pixels" and len(examples) < visualize:
            observations = batch.observations.cpu().numpy()
            support_prediction = objective.decode_prediction(predictions[::2]).float().cpu().numpy()
            for i in range(min(visualize - len(examples), len(batch))):
                examples.append(
                    [
                        observations[i, 0],
                        observations[i, 1],
                        support_prediction[i],
                        observations[i, 2],
                        observations[i, 3],
                        query_prediction[i],
                    ]
                )

    data = {k: np.concatenate(v) for k, v in per_episode.items()}
    order = np.argsort(data["index"])
    data = {k: v[order] for k, v in data.items()}

    # Ground truth for the query pair (slot 2 = input, slot 3 = target).
    true_in = metadata.positions[:, 2]  # (N, frame, ball, xy)
    true_out = metadata.positions[:, 3]
    query_radii = metadata.radii[:, 1]
    step_time = frame_step * FRAME_DT
    true_pos = true_out[:, 1]
    true_vel = (true_out[:, 1] - true_out[:, 0]) / step_time
    trackable = _fully_inside(true_out[:, 0], query_radii) & _fully_inside(true_out[:, 1], query_radii)
    detected = data["pred_visible"] & trackable

    elapsed = (metadata.delta * frame_step * FRAME_DT)[:, None, None]
    input_vel = (true_in[:, 1] - true_in[:, 0]) / step_time
    baselines = {
        "copy": (true_in[:, 1], input_vel),
        "constant_velocity": (true_in[:, 1] + input_vel * elapsed, input_vel),
    }

    def errors(position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.linalg.norm(position - true_pos, axis=-1),
            np.linalg.norm(velocity - true_vel, axis=-1),
        )

    model_pos_err, model_vel_err = errors(data["pred_pos"], data["pred_vel"])
    baseline_errors = {key: errors(*value) for key, value in baselines.items()}

    def block(mask_episode: np.ndarray) -> dict[str, Any]:
        ball_mask = trackable & mask_episode[:, None]
        model_mask = detected & mask_episode[:, None]
        result = {
            "episodes": int(mask_episode.sum()),
            "query_loss": _summary(data["query_loss"], mask_episode),
            "support_loss": _summary(data["support_loss"], mask_episode),
            "length_matches_delta": _summary(
                (data["length"] == metadata.delta).astype(float), mask_episode
            ),
            "mean_selected_length": _summary(data["length"].astype(float), mask_episode),
            "ball_detection_rate": float(model_mask.sum() / max(ball_mask.sum(), 1)),
            "position_error": _summary(model_pos_err, model_mask),
            "velocity_error": _summary(model_vel_err, model_mask),
        }
        for key, (pos_err, vel_err) in baseline_errors.items():
            result[f"{key}_position_error"] = _summary(pos_err, ball_mask)
            result[f"{key}_velocity_error"] = _summary(vel_err, ball_mask)
        return result

    everything = np.ones(len(metadata.delta), dtype=bool)
    collides = metadata.collides[:, 1]
    report: dict[str, Any] = {
        "split": name,
        "path": str(path),
        "num_transitions": transitions,
        "frame_step": frame_step,
        "all": block(everything),
        "query_window_with_collision": block(collides),
        "query_window_without_collision": block(~collides),
        "by_delta": {int(d): block(metadata.delta == d) for d in np.unique(metadata.delta)},
    }

    programs = [
        "-".join(str(c) for c in codes[:length]) for codes, length in zip(data["codes"], data["length"])
    ]
    report["programs"] = {
        "most_common": Counter(programs).most_common(15),
        "with_collision": Counter(p for p, c in zip(programs, collides) if c).most_common(8),
        "without_collision": Counter(p for p, c in zip(programs, collides) if not c).most_common(8),
        "by_delta": {
            int(d): Counter(p for p, dd in zip(programs, metadata.delta) if dd == d).most_common(5)
            for d in np.unique(metadata.delta)
        },
    }
    first_codes = data["codes"][:, 0]
    report["codebook_usage_first_step"] = {
        int(c): int(n) for c, n in zip(*np.unique(first_codes, return_counts=True))
    }

    if examples:
        save_rows(output_dir / f"{name}_examples.png", examples)
        report["examples_png"] = str(output_dir / f"{name}_examples.png")
        report["examples_columns"] = [
            "support x", "support y", "support prediction", "query x", "query y", "query prediction"
        ]
    return report


def _parse_split(text: str) -> tuple[str, Path]:
    name, separator, path = text.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("use --split NAME=PATH")
    return name, Path(path)


def _format_row(name: str, block: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "   n/a" if value is None else f"{value:6.3f}"

    return (
        f"{name:<34}{block['episodes']:>7}  {fmt(block['query_loss'])}  {fmt(block['length_matches_delta'])}"
        f"  {fmt(block['position_error'])}  {fmt(block['velocity_error'])}"
        f"  {fmt(block['constant_velocity_velocity_error'])}  {fmt(block['copy_velocity_error'])}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=_parse_split, action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-transitions", type=int, help="program steps (default: see above)")
    parser.add_argument("--visualize", type=int, default=8, help="example episodes to draw per split")
    parser.add_argument("--cpu", action="store_true")
    arguments = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() and not arguments.cpu else "cpu")
    model = load_neo_checkpoint(arguments.checkpoint, device)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, path in arguments.split:
        reports[name] = evaluate_split(
            model,
            path,
            batch_size=arguments.batch_size,
            num_transitions=arguments.num_transitions,
            device=device,
            visualize=arguments.visualize,
            output_dir=arguments.output_dir,
            name=name,
        )
    (arguments.output_dir / "evaluation.json").write_text(json.dumps(reports, indent=2))

    header = (
        f"{'split / subset':<34}{'eps':>7}  {'qloss':>6}  {'len=Δ':>6}  {'pos_err':>6}  {'vel_err':>6}"
        f"  {'cv_vel':>6}  {'cp_vel':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, report in reports.items():
        print(_format_row(f"{name}", report["all"]))
        print(_format_row(f"{name} / collision in window", report["query_window_with_collision"]))
        print(_format_row(f"{name} / no collision", report["query_window_without_collision"]))
    print(
        "\npos/vel errors are in world units (the world is 10 wide); cv = constant-velocity "
        "reference, cp = copy reference.\nFull report: "
        + str(arguments.output_dir / "evaluation.json")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
