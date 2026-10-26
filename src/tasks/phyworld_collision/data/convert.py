"""Convert a PhyWorld collision HDF5 into one Learning-to-Theorize episode file.

Run it once per split.  Typical set (``k = 2`` frames per step)::

    SRC=data/phyworld/collision_30K.hdf5
    python -m tasks.phyworld_collision.data.convert --source $SRC --videos train \\
        --deltas 1,2,3,4 --episodes 60000 --output data/collision/train.h5
    python -m tasks.phyworld_collision.data.convert --source $SRC --videos test \\
        --deltas 1,2,3,4 --episodes 5000 --output data/collision/id_test.h5
    python -m tasks.phyworld_collision.data.convert --source $SRC --videos test \\
        --deltas 5,6,7,8 --episodes 5000 --output data/collision/length_ood.h5
    python -m tasks.phyworld_collision.data.convert --source collision_eval.hdf5 \\
        --videos all --filter ood-velocity --deltas 1,2,3,4 --episodes 5000 \\
        --output data/collision/velocity_ood.h5

``--videos train/test`` split the source *by video* with a fixed seed, so the
test episodes always come from videos the model never saw.  Support and query
of an episode come from two different videos of the same chunk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from tasks.phyworld_collision.data.episodes import (
    FILTERS,
    EpisodeWriter,
    build_episode,
    classify_videos,
    rendered_frame_source,
    valid_start_range,
    video_split_mask,
)
from tasks.phyworld_collision.data.phyworld_hdf5 import PhyWorldCollisionFile
from tasks.phyworld_collision.physics import downsample_frames


def _parse_deltas(text: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part) for part in text.split(",") if part.strip()}))
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("--deltas needs positive integers, e.g. 1,2,3,4")
    return values


def _allocate(counts: np.ndarray, total: int) -> np.ndarray:
    """Split ``total`` episodes across chunks in proportion to eligible videos."""

    usable = np.where(counts >= 2, counts, 0).astype(np.float64)
    if usable.sum() == 0:
        raise RuntimeError("no chunk has two eligible videos; check --filter and --videos")
    raw = usable / usable.sum() * total
    allocation = np.floor(raw).astype(int)
    remainder = total - allocation.sum()
    for index in np.argsort(-(raw - allocation))[:remainder]:
        allocation[index] += 1
    return allocation


def convert(arguments: argparse.Namespace) -> dict:
    rng = np.random.default_rng(arguments.seed)
    deltas = arguments.deltas
    has_pixels = not arguments.states_only
    with PhyWorldCollisionFile(arguments.source) as source:
        num_frames = None
        chunk_info = []
        for chunk in source.chunks():
            num_frames = chunk.positions.shape[1]
            labels = classify_videos(chunk.init)
            chunk_info.append((chunk.first_index, len(chunk), labels))
        for delta in deltas:
            valid_start_range(delta, arguments.frame_step, num_frames)

        split = video_split_mask(
            source.num_videos,
            arguments.videos,
            test_fraction=arguments.test_fraction,
            split_seed=arguments.split_seed,
        )
        eligible_per_chunk = []
        for first, size, labels in chunk_info:
            mask = FILTERS[arguments.filter](labels) & split[first : first + size]
            eligible_per_chunk.append(np.flatnonzero(mask))
        counts = np.array([len(e) for e in eligible_per_chunk])
        allocation = _allocate(counts, arguments.episodes)

        writer = EpisodeWriter(
            path=arguments.output,
            capacity=arguments.episodes,
            image_size=arguments.image_size,
            frame_step=arguments.frame_step,
            deltas=deltas,
            has_pixels=has_pixels,
            attributes={
                "source": str(Path(arguments.source).resolve()),
                "videos": arguments.videos,
                "filter": arguments.filter,
                "render": arguments.render if has_pixels else "none",
                "split_seed": arguments.split_seed,
                "test_fraction": arguments.test_fraction,
                "seed": arguments.seed,
            },
        )
        label_counts: dict[str, int] = {}
        collisions = 0
        try:
            for chunk, eligible, budget in zip(source.chunks(), eligible_per_chunk, allocation):
                if budget == 0:
                    continue
                frame_source = None
                if has_pixels and arguments.render == "positions":
                    frame_source = rendered_frame_source(
                        chunk.positions, chunk.radii, arguments.image_size
                    )
                elif has_pixels:
                    decoded: dict[int, np.ndarray] = {}

                    def frame_source(video: int, frames: np.ndarray, chunk=chunk, decoded=decoded):
                        if video not in decoded:
                            decoded[video] = downsample_frames(
                                source.decode_video(chunk, video), arguments.image_size
                            )
                        return decoded[video][frames]

                labels = classify_videos(chunk.init)
                for _ in range(budget):
                    support, query = rng.choice(eligible, size=2, replace=False)
                    record = build_episode(
                        rng,
                        support_video=int(support),
                        query_video=int(query),
                        delta=int(rng.choice(deltas)),
                        frame_step=arguments.frame_step,
                        positions=chunk.positions,
                        init=chunk.init,
                        first_index=chunk.first_index,
                        frame_source=frame_source,
                        collision_fraction=arguments.collision_fraction,
                    )
                    writer.append(record)
                    collisions += int(record.collides[1])
                    for video in (support, query):
                        label_counts[labels[video]] = label_counts.get(labels[video], 0) + 1
                print(f"  {writer.count}/{arguments.episodes} episodes", flush=True)
        finally:
            writer.close()
    summary = {
        "output": str(arguments.output),
        "episodes": writer.count,
        "eligible_videos": int(counts.sum()),
        "deltas": list(deltas),
        "frame_step": arguments.frame_step,
        "image_size": arguments.image_size if has_pixels else None,
        "query_windows_with_collision": collisions / max(writer.count, 1),
        "video_labels": label_counts,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="PhyWorld collision HDF5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--deltas", type=_parse_deltas, default=(1, 2, 3, 4))
    parser.add_argument("--frame-step", type=int, default=2, help="raw frames per step (k)")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--render",
        choices=("positions", "video"),
        default="positions",
        help="draw frames from stored positions (default) or decode the MP4s",
    )
    parser.add_argument("--states-only", action="store_true", help="skip pixels (state-vector runs)")
    parser.add_argument("--videos", choices=("train", "test", "all"), default="all")
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--filter", choices=tuple(FILTERS), default="all")
    parser.add_argument(
        "--collision-fraction",
        type=float,
        help="probability that a window contains the contact (default: natural rate)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.frame_step < 1:
        raise SystemExit("--frame-step must be positive")
    summary = convert(arguments)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
