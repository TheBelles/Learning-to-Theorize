"""Turn collision videos into Learning-to-Theorize episodes.

Task definition ("advance time by delta steps")
-----------------------------------------------
* An **observation** at frame ``t`` is two frames, ``t - k`` and ``t``, where
  ``k = frame_step``.  Two frames make the state Markov: they show positions
  *and* velocities.  Pixels are stacked on the channel axis, ``(S, S, 6)``;
  the state version is the 10-number vector of :func:`physics.state_vector`.
* A **pair** is ``x = obs(t)`` and ``y = obs(t + delta * k)``.
* An **episode** is a support pair from video A and a query pair from a
  different video B with the **same delta**.  NEO infers a program on the
  support pair and must reuse it on the query pair, so the only thing the
  two pairs share is "advance time by delta steps": the physics.

Episode file layout (one HDF5 per split)
----------------------------------------
Root attributes: ``format``, ``dataset_length``, ``image_size``,
``frame_step``, ``deltas``, ``has_pixels``, ``source``, ``split``.

==================  ======================  ==========================================
dataset             shape / dtype           meaning
==================  ======================  ==========================================
grids               (N, 4, S, S, 6) uint8   x_support, y_support, x_query, y_query
states              (N, 4, 10) float32      same four observations as state vectors
positions           (N, 4, 2, 2, 2) f32     true positions: obs, frame (t-k, t), ball, xy
radii               (N, 2, 2) f32           support/query video, ball
init                (N, 2, 4) f32           r1, r2, v1, v2 of the support/query video
delta               (N,) int16              steps advanced (frames = delta * frame_step)
collides            (N, 2) bool             contact inside the support/query window
video_index         (N, 2) int64            global video index in the source file
start_frame         (N, 2) int16            t of the support/query input observation
==================  ======================  ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tasks.phyworld_collision.physics import (
    ID_RADIUS_RANGE,
    ID_VELOCITY_RANGE,
    STATE_DIM,
    collision_frame,
    render_frames,
    state_vector,
    window_contains_collision,
)


FORMAT = "l2t-phyworld-collision-episodes-v1"
CHANNELS = 6
OBSERVATIONS_PER_EPISODE = 4


# ---------------------------------------------------------------------------
# Video selection
# ---------------------------------------------------------------------------


def classify_videos(init: np.ndarray, *, tol: float = 1e-6) -> np.ndarray:
    """Label each ``(r1, r2, v1, v2)`` row as id / ood-velocity / ood-radius / ood-both."""

    init = np.asarray(init, dtype=np.float64)
    v_ok = np.all(
        (init[:, 2:] >= ID_VELOCITY_RANGE[0] - tol) & (init[:, 2:] <= ID_VELOCITY_RANGE[1] + tol),
        axis=1,
    )
    r_ok = np.all(
        (init[:, :2] >= ID_RADIUS_RANGE[0] - tol) & (init[:, :2] <= ID_RADIUS_RANGE[1] + tol),
        axis=1,
    )
    labels = np.full(len(init), "ood-both", dtype=object)
    labels[v_ok & r_ok] = "id"
    labels[~v_ok & r_ok] = "ood-velocity"
    labels[v_ok & ~r_ok] = "ood-radius"
    return labels


FILTERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "all": lambda labels: np.ones(len(labels), dtype=bool),
    "id": lambda labels: labels == "id",
    "ood": lambda labels: labels != "id",
    "ood-velocity": lambda labels: labels == "ood-velocity",
    "ood-radius": lambda labels: labels == "ood-radius",
}


def video_split_mask(
    num_videos: int, which: str, *, test_fraction: float, split_seed: int
) -> np.ndarray:
    """Deterministic train/test split *by video*, so test videos are never trained on."""

    if which == "all":
        return np.ones(num_videos, dtype=bool)
    order = np.random.default_rng(split_seed).permutation(num_videos)
    is_test = np.zeros(num_videos, dtype=bool)
    is_test[order[: int(round(test_fraction * num_videos))]] = True
    if which == "test":
        return is_test
    if which == "train":
        return ~is_test
    raise ValueError(f"unknown video split {which!r}")


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def valid_start_range(delta: int, frame_step: int, num_frames: int) -> tuple[int, int]:
    """Inclusive range of input frames ``t`` with room for ``t - k`` and ``t + delta * k``."""

    low = frame_step
    high = num_frames - 1 - delta * frame_step
    if high < low:
        raise ValueError(
            f"delta={delta} with frame_step={frame_step} does not fit in {num_frames} frames"
        )
    return low, high


def sample_start(
    rng: np.random.Generator,
    *,
    delta: int,
    frame_step: int,
    num_frames: int,
    contact: int,
    want_collision: bool | None,
) -> int:
    """Pick the input frame ``t``; optionally force the window to (not) contain the contact.

    The window spans frames ``[t - k, t + delta * k]``.  If the requested
    condition cannot be met for this video, a uniform start is returned.
    """

    low, high = valid_start_range(delta, frame_step, num_frames)
    starts = np.arange(low, high + 1)
    if want_collision is not None:
        inside = np.array(
            [
                window_contains_collision(contact, t - frame_step, t + delta * frame_step)
                for t in starts
            ]
        )
        candidates = starts[inside] if want_collision else starts[~inside]
        if candidates.size:
            starts = candidates
    return int(rng.choice(starts))


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def pixel_observation(frames: np.ndarray, t: int, frame_step: int) -> np.ndarray:
    """``(S, S, 6)`` uint8: frame ``t - k`` in channels 0-2, frame ``t`` in 3-5."""

    return np.concatenate([frames[t - frame_step], frames[t]], axis=-1)


@dataclass(slots=True)
class EpisodeRecord:
    """One episode before it is written (all four observations)."""

    grids: np.ndarray | None
    states: np.ndarray
    positions: np.ndarray
    radii: np.ndarray
    init: np.ndarray
    delta: int
    collides: np.ndarray
    video_index: np.ndarray
    start_frame: np.ndarray


FrameSource = Callable[[int, np.ndarray], np.ndarray]
"""``(local_video_index, frame_indices) -> (len(frame_indices), S, S, 3)`` uint8."""


def rendered_frame_source(positions: np.ndarray, radii: np.ndarray, image_size: int) -> FrameSource:
    """Render frames from stored positions (fast, no MP4 artefacts)."""

    def source(video: int, frame_indices: np.ndarray) -> np.ndarray:
        return render_frames(positions[video, frame_indices], radii[video], image_size)

    return source


def build_episode(
    rng: np.random.Generator,
    *,
    support_video: int,
    query_video: int,
    delta: int,
    frame_step: int,
    positions: np.ndarray,
    init: np.ndarray,
    first_index: int,
    frame_source: FrameSource | None,
    collision_fraction: float | None,
) -> EpisodeRecord:
    """Assemble one support/query episode from two videos of the same chunk."""

    num_frames = positions.shape[1]
    videos = (support_video, query_video)
    grids, states, true_positions, starts, collides = [], [], [], [], []
    for video in videos:
        contact = collision_frame(positions[video], init[video, :2])
        want = None
        if collision_fraction is not None:
            want = bool(rng.random() < collision_fraction)
        t = sample_start(
            rng,
            delta=delta,
            frame_step=frame_step,
            num_frames=num_frames,
            contact=contact,
            want_collision=want,
        )
        t_out = t + delta * frame_step
        starts.append(t)
        collides.append(window_contains_collision(contact, t - frame_step, t_out))
        for end in (t, t_out):
            states.append(state_vector(positions[video], init[video, :2], end, frame_step))
            true_positions.append(positions[video, [end - frame_step, end]])
        if frame_source is not None:
            needed = np.array([t - frame_step, t, t_out - frame_step, t_out])
            frames = frame_source(video, needed)
            grids.append(np.concatenate([frames[0], frames[1]], axis=-1))
            grids.append(np.concatenate([frames[2], frames[3]], axis=-1))
    return EpisodeRecord(
        grids=np.stack(grids).astype(np.uint8) if frame_source is not None else None,
        states=np.stack(states).astype(np.float32),
        positions=np.stack(true_positions).astype(np.float32),
        radii=init[list(videos), :2].astype(np.float32),
        init=init[list(videos)].astype(np.float32),
        delta=int(delta),
        collides=np.asarray(collides, dtype=bool),
        video_index=np.asarray([first_index + v for v in videos], dtype=np.int64),
        start_frame=np.asarray(starts, dtype=np.int16),
    )


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("h5py is required to write episode files") from error
    return h5py


@dataclass
class EpisodeWriter:
    """Append episodes to a new HDF5 file (see the module docstring for the layout)."""

    path: Path
    capacity: int
    image_size: int
    frame_step: int
    deltas: tuple[int, ...]
    has_pixels: bool
    attributes: dict[str, Any] = field(default_factory=dict)
    compression_level: int = 4

    def __post_init__(self) -> None:
        h5py = _h5py()
        self.path = Path(self.path)
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w")
        n, s = self.capacity, self.image_size
        create = self._file.create_dataset
        if self.has_pixels:
            create(
                "grids",
                shape=(n, 4, s, s, CHANNELS),
                maxshape=(None, 4, s, s, CHANNELS),
                dtype=np.uint8,
                chunks=(1, 4, s, s, CHANNELS),
                compression="gzip",
                compression_opts=self.compression_level,
            )
        create("states", shape=(n, 4, STATE_DIM), maxshape=(None, 4, STATE_DIM), dtype=np.float32)
        create("positions", shape=(n, 4, 2, 2, 2), maxshape=(None, 4, 2, 2, 2), dtype=np.float32)
        create("radii", shape=(n, 2, 2), maxshape=(None, 2, 2), dtype=np.float32)
        create("init", shape=(n, 2, 4), maxshape=(None, 2, 4), dtype=np.float32)
        create("delta", shape=(n,), maxshape=(None,), dtype=np.int16)
        create("collides", shape=(n, 2), maxshape=(None, 2), dtype=bool)
        create("video_index", shape=(n, 2), maxshape=(None, 2), dtype=np.int64)
        create("start_frame", shape=(n, 2), maxshape=(None, 2), dtype=np.int16)
        self.count = 0

    def append(self, record: EpisodeRecord) -> None:
        if self.count >= self.capacity:
            raise RuntimeError("episode writer is full")
        i = self.count
        if self.has_pixels:
            if record.grids is None:
                raise ValueError("pixel episode file needs grids")
            self._file["grids"][i] = record.grids
        self._file["states"][i] = record.states
        self._file["positions"][i] = record.positions
        self._file["radii"][i] = record.radii
        self._file["init"][i] = record.init
        self._file["delta"][i] = record.delta
        self._file["collides"][i] = record.collides
        self._file["video_index"][i] = record.video_index
        self._file["start_frame"][i] = record.start_frame
        self.count += 1

    def close(self) -> None:
        if self._file is None:
            return
        for name in list(self._file.keys()):
            self._file[name].resize(self.count, axis=0)
        attrs = self._file.attrs
        attrs["format"] = FORMAT
        attrs["dataset_length"] = self.count
        attrs["image_size"] = self.image_size
        attrs["frame_step"] = self.frame_step
        attrs["deltas"] = np.asarray(self.deltas, dtype=np.int64)
        attrs["has_pixels"] = self.has_pixels
        attrs["state_dim"] = STATE_DIM
        for key, value in self.attributes.items():
            attrs[key] = value
        self._file.close()
        self._file = None


def read_attributes(path: str | Path) -> dict[str, Any]:
    """Root attributes of an episode file, validated."""

    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        attrs = {key: handle.attrs[key] for key in handle.attrs}
    if attrs.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file (format={attrs.get('format')!r})")
    return attrs


__all__ = [
    "CHANNELS",
    "EpisodeRecord",
    "EpisodeWriter",
    "FILTERS",
    "FORMAT",
    "build_episode",
    "classify_videos",
    "pixel_observation",
    "read_attributes",
    "rendered_frame_source",
    "sample_start",
    "valid_start_range",
    "video_split_mask",
]
