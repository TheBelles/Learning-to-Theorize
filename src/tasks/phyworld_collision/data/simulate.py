"""Generate PhyWorld-format collision files without Box2D.

Useful for (a) a quick smoke test before downloading the real data and
(b) building OOD evaluation sets with exactly the ranges you want.  The
parameter grids follow PhyWorld's ``two_balls_collision.py``:

* ``id-30k``: v1, v2 on 21 values in [1, 4]; r1, r2 on 8 values in [0.5, 1.5]
  (28,224 combinations, the "30K" training set);
* ``id-eval``: 15 random speeds and 5 random radii in the ID ranges;
* ``ood-velocity-low`` / ``ood-velocity-high``: one or both speeds in
  [0, 0.8] / [4.5, 6.0], radii in distribution (PhyWorld levels 2 and 4);
* ``ood-radius-low`` / ``ood-radius-high``: one or both radii in
  [0.3, 0.4] / [1.7, 2.0], speeds in distribution (levels 1 and 3);
* ``tiny``: 4 speeds x 3 radii, for tests.

The motion is the closed-form 1D collision of :mod:`..physics`; contacts
slower than Box2D's 1 unit/s velocity threshold are inelastic, as in Box2D.

Example::

    python -m tasks.phyworld_collision.data.simulate --preset ood-velocity-high \\
        --output data/phyworld/collision_ood_velocity_high.hdf5
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Sequence

import numpy as np

from tasks.phyworld_collision.data.phyworld_hdf5 import write_phyworld_file
from tasks.phyworld_collision.physics import (
    ID_RADIUS_RANGE,
    ID_VELOCITY_RANGE,
    sample_initial_positions,
    simulate_collision,
)


def _grid(r1s, r2s, v1s, v2s) -> list[tuple[float, float, float, float]]:
    return [tuple(map(float, c)) for c in itertools.product(r1s, r2s, v1s, v2s)]


def preset_combinations(name: str, rng: np.random.Generator) -> list[tuple[float, float, float, float]]:
    """``(r1, r2, v1, v2)`` combinations of a named preset."""

    v_lo, v_hi = ID_VELOCITY_RANGE
    r_lo, r_hi = ID_RADIUS_RANGE
    v_id = np.linspace(v_lo, v_hi, 16)[::2]
    r_id = np.linspace(r_lo, r_hi, 6)
    if name == "id-30k":
        v, r = np.linspace(v_lo, v_hi, 21), np.linspace(r_lo, r_hi, 8)
        return _grid(r, r, v, v)
    if name == "id-eval":
        v = np.sort(rng.uniform(v_lo, v_hi, 15))
        r = np.sort(rng.uniform(r_lo, r_hi, 5))
        return _grid(r, r, v, v)
    if name in ("ood-velocity-low", "ood-velocity-high"):
        v_ood = np.linspace(0.0, 0.8, 11) if name.endswith("low") else np.linspace(4.5, 6.0, 6)
        return _grid(r_id, r_id, v_ood, v_id) + _grid(r_id, r_id, v_ood, v_ood)
    if name in ("ood-radius-low", "ood-radius-high"):
        r_ood = np.linspace(0.3, 0.4, 11) if name.endswith("low") else np.linspace(1.7, 2.0, 6)
        return _grid(r_ood, r_id, v_id, v_id) + _grid(r_ood, r_ood, v_id, v_id)
    if name == "tiny":
        v, r = np.linspace(v_lo, v_hi, 4), np.linspace(r_lo, r_hi, 3)
        return _grid(r, r, v, v)
    raise ValueError(f"unknown preset {name!r}")


PRESETS = (
    "id-30k",
    "id-eval",
    "ood-velocity-low",
    "ood-velocity-high",
    "ood-radius-low",
    "ood-radius-high",
    "tiny",
)


def simulate_videos(
    combinations: Sequence[tuple[float, float, float, float]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate each combination; drops scenes PhyWorld would also drop."""

    positions, init = [], []
    for r1, r2, v1, v2 in combinations:
        placement = sample_initial_positions(r1, r2, v1, v2, rng)
        if placement is None:
            continue
        x1, x2, height = placement
        positions.append(simulate_collision(r1, r2, v1, v2, x1, x2, height))
        init.append((r1, r2, v1, v2))
    if not positions:
        raise RuntimeError("no scene could be placed")
    return np.stack(positions), np.asarray(init, dtype=np.float64)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, help="random subset of the preset's combinations")
    parser.add_argument("--with-video", action="store_true", help="also store 256x256 MP4s (needs imageio-ffmpeg)")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args(argv)

    rng = np.random.default_rng(arguments.seed)
    combinations = preset_combinations(arguments.preset, rng)
    order = rng.permutation(len(combinations))
    if arguments.max_videos is not None:
        order = order[: arguments.max_videos]
    combinations = [combinations[i] for i in order]
    positions, init = simulate_videos(combinations, rng)
    write_phyworld_file(
        arguments.output,
        positions,
        init,
        chunk_size=arguments.chunk_size,
        with_video=arguments.with_video,
    )
    print(f"wrote {len(positions)} videos ({arguments.preset}) to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
