"""PhyWorld two-ball collision: constants, rendering, simulation and ball tracking.

Everything here is NumPy-only so it can be used by the data tools, the
evaluation and the tests without PyTorch.

Conventions copied from PhyWorld ``id_ood_data/two_balls_collision.py``:

* the world is ``WORLD_SCALE = 10`` units wide and high;
* Box2D runs at ``timestep = 0.01`` and every ``STRIDE = 10``-th step is kept,
  so consecutive frames are ``FRAME_DT = 0.1`` seconds apart;
* each video has ``NUM_FRAMES = 32`` frames;
* ball 1 is red, starts on the left and moves right with speed ``v1``;
  ball 2 is blue, starts on the right and moves left with speed ``v2``;
* both balls have density 1, so their masses are proportional to ``r**2``;
* positions are stored as ``(frames, ball, xy)`` in world units and the
  rendered image is flipped vertically (``frames[:, ::-1]``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


WORLD_SCALE = 10.0
TIMESTEP = 0.01
STRIDE = 10
FRAME_DT = TIMESTEP * STRIDE
NUM_FRAMES = 32
MIN_PRE_FRAMES = 8
MIN_POST_FRAMES = 8

#: In-distribution ranges of the PhyWorld collision training data.
ID_VELOCITY_RANGE = (1.0, 4.0)
ID_RADIUS_RANGE = (0.5, 1.5)

#: Box2D treats contacts slower than this relative speed as inelastic.
BOX2D_VELOCITY_THRESHOLD = 1.0

BALL_COLORS = np.array([(255, 0, 0), (0, 0, 255)], dtype=np.float64)

#: Normalisation of the state vector (see :func:`state_vector`).
POSITION_SCALE = WORLD_SCALE
VELOCITY_SCALE = 4.0
RADIUS_SCALE = 1.5
STATE_DIM = 10


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_frames(
    positions: np.ndarray,
    radii: np.ndarray,
    image_size: int,
    *,
    edge_width: float = 0.03,
) -> np.ndarray:
    """Render ``(T, 2, 2)`` world positions as ``(T, S, S, 3)`` uint8 frames.

    This is PhyWorld's ``vectorized_draw_balls`` (soft-edged discs on white,
    ball 2 drawn over ball 1) followed by its vertical flip.
    """

    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim == 2:
        positions = positions[None]
    if positions.shape[1:] != (2, 2):
        raise ValueError(f"positions must be (T, 2, 2), got {positions.shape}")
    normalized_positions = positions / WORLD_SCALE
    normalized_radii = np.asarray(radii, dtype=np.float64) / WORLD_SCALE

    grid = np.linspace(0.0, 1.0, image_size)
    xs, ys = np.meshgrid(grid, grid)
    frames = np.full((positions.shape[0], image_size, image_size, 3), 255.0)
    for ball in range(2):
        dx = xs[None] - normalized_positions[:, ball, 0, None, None]
        dy = ys[None] - normalized_positions[:, ball, 1, None, None]
        radius_squared = normalized_radii[ball] ** 2
        mask = np.clip(
            (radius_squared - (dx**2 + dy**2)) / (radius_squared * edge_width), 0.0, 1.0
        )[..., None]
        frames = (1.0 - mask) * frames + mask * BALL_COLORS[ball]
    frames = frames.astype(np.uint8)
    return np.ascontiguousarray(frames[:, ::-1])


def downsample_frames(frames: np.ndarray, image_size: int) -> np.ndarray:
    """Area-downsample ``(T, H, W, 3)`` uint8 frames to ``image_size``."""

    frames = np.asarray(frames)
    height, width = frames.shape[1:3]
    if height == image_size and width == image_size:
        return frames.astype(np.uint8, copy=False)
    if height % image_size == 0 and width % image_size == 0:
        fy, fx = height // image_size, width // image_size
        blocks = frames.reshape(
            frames.shape[0], image_size, fy, image_size, fx, frames.shape[3]
        )
        return np.round(blocks.mean(axis=(2, 4))).astype(np.uint8)
    from PIL import Image

    return np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize((image_size, image_size), Image.BOX)
            )
            for frame in frames
        ]
    )


# ---------------------------------------------------------------------------
# Analytic simulation (PhyWorld-compatible, no Box2D needed)
# ---------------------------------------------------------------------------


def masses(r1: float, r2: float) -> tuple[float, float]:
    """Disc masses at density 1."""

    return np.pi * r1**2, np.pi * r2**2


def post_collision_velocities(
    r1: float, r2: float, u1: float, u2: float, *, inelastic_threshold: float | None
) -> tuple[float, float]:
    """Velocities after a head-on contact; ``u1``/``u2`` are signed x-velocities."""

    m1, m2 = masses(r1, r2)
    if inelastic_threshold is not None and (u1 - u2) < inelastic_threshold:
        common = (m1 * u1 + m2 * u2) / (m1 + m2)
        return common, common
    w1 = ((m1 - m2) * u1 + 2.0 * m2 * u2) / (m1 + m2)
    w2 = ((m2 - m1) * u2 + 2.0 * m1 * u1) / (m1 + m2)
    return w1, w2


def sample_initial_positions(
    r1: float,
    r2: float,
    v1: float,
    v2: float,
    rng: np.random.Generator,
    *,
    eps: float = 0.05,
) -> tuple[float, float, float] | None:
    """Port of PhyWorld's placement logic; returns ``(x1, x2, height)``.

    The original's control flow is kept on purpose, including its quirks: the
    ``r1 > 2 r2`` branch is followed by an independent ``if/else`` (so its
    else-branch overrides it), ``max_pre_dist`` is reused across retries, and a
    scene that needed 10 or more retries is discarded (returns ``None``).
    """

    def uniform(low: float, high: float) -> float:
        return float(rng.random() * (high - low) + low)

    frame_time = TIMESTEP * STRIDE
    min_pre_dist = MIN_PRE_FRAMES * (v1 + v2) * frame_time + r1 + r2
    max_pre_dist = (NUM_FRAMES - MIN_POST_FRAMES) * frame_time * (v1 + v2) + r1 + r2
    x1 = x2 = 0.0
    tries = 0
    while tries < 50:
        if r1 > 2 * r2:
            max_pre_dist = min(min_pre_dist * 1.25, WORLD_SCALE - r1 - r2)
            dist = uniform(min_pre_dist, max_pre_dist)
            x1_left = r1
            x1_right = x1_left + eps * WORLD_SCALE
        if r2 > 2 * r1:
            max_pre_dist = min(min_pre_dist * 1.25, WORLD_SCALE - r1 - r2)
            dist = uniform(min_pre_dist, max_pre_dist)
            x1_right = WORLD_SCALE - r2 - dist
            x1_left = x1_right - eps * WORLD_SCALE
        else:
            max_pre_dist = min(max_pre_dist, WORLD_SCALE - r1 - r2)
            dist = uniform(min_pre_dist, max_pre_dist)
            x1_left = r1
            x1_right = WORLD_SCALE - r2 - dist
        x1 = max(uniform(x1_left, x1_right), r1)
        x2 = min(x1 + dist, WORLD_SCALE - r2)
        pre_frames = (
            (x2 - x1 - r1 - r2) / (v1 + v2) / frame_time
            if v1 + v2 > 0
            else MIN_PRE_FRAMES + 1
        )
        if pre_frames < MIN_PRE_FRAMES:
            tries += 1
            continue
        break
    if tries >= 10:
        return None
    max_r = max(r1, r2)
    height = uniform(max_r + eps * WORLD_SCALE, (1 - eps) * WORLD_SCALE - max_r)
    return float(x1), float(x2), float(height)


def simulate_collision(
    r1: float,
    r2: float,
    v1: float,
    v2: float,
    x1: float,
    x2: float,
    height: float,
    *,
    num_frames: int = NUM_FRAMES,
    inelastic_threshold: float | None = BOX2D_VELOCITY_THRESHOLD,
) -> np.ndarray:
    """Closed-form 1D collision sampled like PhyWorld; returns ``(T, 2, 2)``.

    Box2D records positions after steps ``0, 10, 20, ...``, i.e. at times
    ``(10 k + 1) * 0.01``.  Between contacts the balls move uniformly; at
    contact the velocities change by :func:`post_collision_velocities`.
    """

    times = (STRIDE * np.arange(num_frames) + 1) * TIMESTEP
    u1, u2 = float(v1), -float(v2)
    gap = x2 - x1 - r1 - r2
    closing = u1 - u2
    contact_time = gap / closing if closing > 0 and gap >= 0 else np.inf
    w1, w2 = post_collision_velocities(
        r1, r2, u1, u2, inelastic_threshold=inelastic_threshold
    )

    before = times < contact_time
    after_dt = np.where(before, 0.0, times - contact_time)
    pre_dt = np.where(before, times, contact_time)
    xs1 = x1 + u1 * pre_dt + w1 * after_dt
    xs2 = x2 + u2 * pre_dt + w2 * after_dt

    positions = np.empty((num_frames, 2, 2), dtype=np.float64)
    positions[:, 0, 0] = xs1
    positions[:, 1, 0] = xs2
    positions[:, :, 1] = height
    return positions


# ---------------------------------------------------------------------------
# Events and state vectors
# ---------------------------------------------------------------------------


def collision_frame(positions: np.ndarray, radii: np.ndarray, *, tol: float = 1e-4) -> int:
    """Index ``k`` such that the contact happens between frames ``k`` and ``k+1``.

    Detected as the first change of ball 1's per-frame displacement; returns
    ``-1`` when the balls never touch inside the video.
    """

    displacement = np.diff(np.asarray(positions)[:, 0, 0])
    if displacement.size < 2:
        return -1
    changed = np.abs(displacement - displacement[0]) > tol
    if not changed.any():
        return -1
    return int(np.argmax(changed))


def window_contains_collision(contact: int, start: int, end: int) -> bool:
    """Whether a contact between frames ``contact`` and ``contact+1`` lies in ``[start, end]``."""

    return contact >= 0 and start <= contact < end


def state_vector(
    positions: np.ndarray, radii: np.ndarray, t: int, frame_step: int
) -> np.ndarray:
    """Normalised ``(10,)`` state of an observation ending at frame ``t``.

    Per ball: ``x, y`` (/10), ``vx, vy`` (finite difference over
    ``frame_step`` frames, /4 units per second) and radius (/1.5).
    """

    current = positions[t]
    previous = positions[t - frame_step]
    velocity = (current - previous) / (frame_step * FRAME_DT)
    values = []
    for ball in range(2):
        values.extend(current[ball] / POSITION_SCALE)
        values.extend(velocity[ball] / VELOCITY_SCALE)
        values.append(radii[ball] / RADIUS_SCALE)
    return np.asarray(values, dtype=np.float32)


def state_to_physical(states: np.ndarray) -> dict[str, np.ndarray]:
    """Invert :func:`state_vector` on ``(..., 10)`` arrays."""

    states = np.asarray(states, dtype=np.float64).reshape(*np.shape(states)[:-1], 2, 5)
    return {
        "position": states[..., 0:2] * POSITION_SCALE,
        "velocity": states[..., 2:4] * VELOCITY_SCALE,
        "radius": states[..., 4] * RADIUS_SCALE,
    }


# ---------------------------------------------------------------------------
# Ball tracking in rendered images
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BallTrack:
    """Centroids (world units) and visibility of both balls in a batch of frames."""

    centers: np.ndarray  # (N, 2, 2): frame, ball, xy
    visible: np.ndarray  # (N, 2) bool


def ball_centroids(frames: np.ndarray, *, min_mass: float = 1.0) -> BallTrack:
    """Locate the red and blue balls in ``(N, H, W, 3)`` frames.

    ``frames`` may be uint8 or floats in ``[0, 1]``.  The weight of a pixel is
    how strongly it shows one ball's colour (e.g. red: ``R - max(G, B)``), so
    anti-aliased edges and blurry predictions still give sub-pixel centroids.
    A ball with total weight below ``min_mass`` pixels is reported invisible.
    """

    frames = np.asarray(frames, dtype=np.float64)
    if frames.max(initial=0.0) > 1.0:
        frames = frames / 255.0
    if frames.ndim == 3:
        frames = frames[None]
    n, height, width, _ = frames.shape
    red, green, blue = frames[..., 0], frames[..., 1], frames[..., 2]
    weights = np.stack(
        [
            np.clip(red - np.maximum(green, blue), 0.0, 1.0),
            np.clip(blue - np.maximum(red, green), 0.0, 1.0),
        ],
        axis=1,
    )  # (N, 2, H, W)
    rows = np.arange(height, dtype=np.float64)
    cols = np.arange(width, dtype=np.float64)
    mass = weights.sum(axis=(2, 3))
    safe = np.maximum(mass, 1e-12)
    col_center = (weights.sum(axis=2) * cols).sum(axis=-1) / safe
    row_center = (weights.sum(axis=3) * rows).sum(axis=-1) / safe
    # Undo the vertical flip and map pixel centres onto PhyWorld's linspace grid.
    x = col_center / (width - 1) * WORLD_SCALE
    y = (height - 1 - row_center) / (height - 1) * WORLD_SCALE
    centers = np.stack([x, y], axis=-1)
    visible = mass >= min_mass
    centers[~visible] = np.nan
    return BallTrack(centers=centers, visible=visible)


__all__ = [
    "BALL_COLORS",
    "BOX2D_VELOCITY_THRESHOLD",
    "BallTrack",
    "FRAME_DT",
    "ID_RADIUS_RANGE",
    "ID_VELOCITY_RANGE",
    "NUM_FRAMES",
    "STATE_DIM",
    "WORLD_SCALE",
    "ball_centroids",
    "collision_frame",
    "downsample_frames",
    "masses",
    "post_collision_velocities",
    "render_frames",
    "sample_initial_positions",
    "simulate_collision",
    "state_to_physical",
    "state_vector",
    "window_contains_collision",
]
