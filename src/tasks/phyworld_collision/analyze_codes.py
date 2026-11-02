"""What do NEO's step codes encode, and does a code mismatch explain transfer failure?

::

    python -m tasks.phyworld_collision.analyze_codes \\
        --checkpoint runs/phyworld_collision/neo_k2/.../checkpoints/best.pt \\
        --split data/phyworld_collision/id_test.h5

For every episode the model is run twice:

* normally: the program is inferred on the support pair and applied to the query;
* swapped: the query pair is used as the support, which gives the code the
  query video would have chosen for itself.

Printed:

1. per code: how often it is chosen and the mean scene parameters of the
   videos that choose it (speeds, radii, mass ratio, collision in window), so
   you can see which kind of scene each code stands for;
2. query error when the support's code equals the query's own code versus when
   it differs. A large difference means the codes are scene classes, and the
   transfer fails whenever the two videos fall into different classes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from tasks.phyworld_collision.data.dataset import EpisodeDataset, collate_episodes, load_metadata
from tasks.phyworld_collision.task import load_neo_checkpoint
from training.runtime import bfloat16_autocast


@torch.no_grad()
def run(model, observations: torch.Tensor, transitions: int, coefficient: float):
    """First code, selected length and query loss for each episode in the batch."""

    with bfloat16_autocast(observations.device):
        output = model(
            observations,
            is_eval=True,
            length_control_coefficient=coefficient,
            num_transitions=transitions,
        )
    _, targets = model.objective.split_batch(observations)
    predictions = output.predictions.float()
    query_loss = model.objective.per_sample_reconstruction_loss(predictions[1::2], targets[1::2])
    return (
        output.action_vq_by_length[0].indices[::2].cpu().numpy(),
        output.selected_lengths[::2].cpu().numpy(),
        query_loss.float().cpu().numpy(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, help="optional JSON with the numbers")
    parser.add_argument("--cpu", action="store_true")
    arguments = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() and not arguments.cpu else "cpu")
    model = load_neo_checkpoint(arguments.checkpoint, device)
    training = model.experiment.training
    dataset = EpisodeDataset(arguments.split, model.observation_config.modality)
    metadata = load_metadata(arguments.split)
    transitions = max(training.max_transition_length, int(metadata.delta.max()))
    loader = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=False, collate_fn=collate_episodes)

    indices, support_code, query_own_code, length, query_loss = [], [], [], [], []
    for batch in loader:
        observations = batch.observations.to(device)
        code, selected, loss = run(model, observations, transitions, training.length_control_end)
        swapped = observations[:, [2, 3, 0, 1]]
        own_code, _, _ = run(model, swapped, transitions, training.length_control_end)
        indices.append(batch.indices.numpy())
        support_code.append(code)
        query_own_code.append(own_code)
        length.append(selected)
        query_loss.append(loss)
    order = np.argsort(np.concatenate(indices))
    support_code = np.concatenate(support_code)[order]
    query_own_code = np.concatenate(query_own_code)[order]
    length = np.concatenate(length)[order]
    query_loss = np.concatenate(query_loss)[order]

    # Scene parameters of the support video (slot 0) and the query video (slot 1).
    init = metadata.init  # (N, 2, 4): r1, r2, v1, v2
    collides = metadata.collides

    def features(video: int) -> dict[str, np.ndarray]:
        r1, r2, v1, v2 = (init[:, video, i] for i in range(4))
        return {
            "v1": v1,
            "v2": v2,
            "v1+v2": v1 + v2,
            "r1": r1,
            "r2": r2,
            "m1/m2": (r1 / r2) ** 2,
            "collision in window": collides[:, video].astype(float),
        }

    # Pool support videos (with the code they chose) and query videos (with their own code).
    codes = np.concatenate([support_code, query_own_code])
    pooled = {k: np.concatenate([features(0)[k], features(1)[k]]) for k in features(0)}

    report: dict = {"checkpoint": str(arguments.checkpoint), "split": str(arguments.split), "codes": {}}
    print(f"\n1. What each first-step code stands for ({len(codes)} videos)\n")
    header = f"{'code':>4} {'share':>6} " + " ".join(f"{k:>10}" for k in pooled)
    print(header)
    print("-" * len(header))
    for code in np.unique(codes):
        mask = codes == code
        means = {k: float(v[mask].mean()) for k, v in pooled.items()}
        report["codes"][int(code)] = {"share": float(mask.mean()), **means}
        print(f"{int(code):>4} {mask.mean():>6.1%} " + " ".join(f"{means[k]:>10.3f}" for k in pooled))
    print(f"{'all':>4} {'':>6} " + " ".join(f"{v.mean():>10.3f}" for v in pooled.values()))

    same = support_code == query_own_code
    length_ok = length == metadata.delta
    rows = {
        "support code = query's own code": same,
        "support code != query's own code": ~same,
        "  ... and length = delta": ~same & length_ok,
    }
    print("\n2. Query error by code agreement\n")
    print(f"{'subset':<36} {'episodes':>9} {'query loss':>11}")
    for name, mask in rows.items():
        value = float(query_loss[mask].mean()) if mask.any() else float("nan")
        report[name.strip()] = {"episodes": int(mask.sum()), "query_loss": value}
        print(f"{name:<36} {int(mask.sum()):>9} {value:>11.4f}")
    ratio = query_loss[~same].mean() / max(query_loss[same].mean(), 1e-12) if (~same).any() and same.any() else float("nan")
    print(f"\nmismatch / match error ratio: {ratio:.2f}")

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
