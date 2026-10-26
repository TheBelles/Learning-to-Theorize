"""Read (and write) PhyWorld collision HDF5 files.

PhyWorld's generator merges per-chunk files into one HDF5 with three groups,
each holding one dataset per chunk of up to 1000 videos::

    video_streams/<chunk>     vlen uint8, one MP4 byte stream per video
    position_streams/<chunk>  float  (N, 32, 2, 2)   frame, ball, xy (world units)
    init_streams/<chunk>      float  (N, 4)          r1, r2, v1, v2

The released files on Hugging Face (``magicr/phyworld``) are not documented
beyond this, so the reader also accepts the three names as plain datasets and
finds them in nested groups (e.g. one group per OOD level).  Run
``python -m tasks.phyworld_collision.data.phyworld_hdf5 FILE`` to print the
layout of a file before converting it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from tasks.phyworld_collision.physics import NUM_FRAMES, render_frames


POSITIONS = "position_streams"
INIT = "init_streams"
VIDEOS = "video_streams"


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("h5py is required to read PhyWorld files") from error
    return h5py


@dataclass(frozen=True, slots=True)
class Chunk:
    """One block of videos from a PhyWorld file, loaded into memory."""

    collection: str
    name: str
    first_index: int  # global index of the first video in this chunk
    positions: np.ndarray  # (N, T, 2, 2) float64
    init: np.ndarray  # (N, 4) float64: r1, r2, v1, v2
    has_video: bool

    def __len__(self) -> int:
        return self.positions.shape[0]

    @property
    def radii(self) -> np.ndarray:
        return self.init[:, :2]


def _collections(handle: Any) -> list[str]:
    """Paths of groups that contain both ``position_streams`` and ``init_streams``."""

    found: list[str] = []

    def visit(path: str, group: Any) -> None:
        if POSITIONS in group and INIT in group:
            found.append(path)
            return
        for key in sorted(group.keys()):
            item = group[key]
            if hasattr(item, "keys"):
                visit(f"{path}/{key}" if path else key, item)

    visit("", handle)
    if not found:
        raise ValueError(
            f"no group with {POSITIONS!r} and {INIT!r} found; run the inspector "
            "(python -m tasks.phyworld_collision.data.phyworld_hdf5 FILE) to see the layout"
        )
    return found


def _chunk_names(node: Any) -> list[str | None]:
    """Chunk dataset names of a stream group, or ``[None]`` for a plain dataset."""

    if hasattr(node, "keys"):
        return sorted(node.keys())
    return [None]


def _read(node: Any, name: str | None) -> Any:
    return node if name is None else node[name]


class PhyWorldCollisionFile:
    """Iterate over the videos of one PhyWorld collision file, chunk by chunk."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve(strict=True)
        h5py = _h5py()
        self._handle = h5py.File(self.path, "r")
        self.collections = _collections(self._handle)
        self._index: list[tuple[str, str | None, int, int]] = []
        total = 0
        for collection in self.collections:
            group = self._handle[collection] if collection else self._handle
            for name in _chunk_names(group[POSITIONS]):
                count = int(_read(group[POSITIONS], name).shape[0])
                self._index.append((collection, name, total, count))
                total += count
        self.num_videos = total

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "PhyWorldCollisionFile":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def chunks(self) -> Iterator[Chunk]:
        for collection, name, first, _ in self._index:
            group = self._handle[collection] if collection else self._handle
            positions = np.asarray(_read(group[POSITIONS], name), dtype=np.float64)
            init = np.asarray(_read(group[INIT], name), dtype=np.float64)
            if positions.ndim != 4 or positions.shape[2:] != (2, 2):
                raise ValueError(
                    f"{collection}/{POSITIONS}/{name}: expected (N, T, 2, 2), got {positions.shape}"
                )
            if init.shape != (positions.shape[0], 4):
                raise ValueError(
                    f"{collection}/{INIT}/{name}: expected ({positions.shape[0]}, 4), got {init.shape}"
                )
            yield Chunk(
                collection=collection,
                name=name or "",
                first_index=first,
                positions=positions,
                init=init,
                has_video=VIDEOS in group,
            )

    def video_bytes(self, chunk: Chunk, index: int) -> bytes:
        group = self._handle[chunk.collection] if chunk.collection else self._handle
        if VIDEOS not in group:
            raise ValueError(f"{self.path} has no {VIDEOS}; use --render positions")
        stream = _read(group[VIDEOS], chunk.name or None)[index]
        return np.asarray(stream, dtype=np.uint8).tobytes()

    def decode_video(self, chunk: Chunk, index: int) -> np.ndarray:
        """Decode one MP4 into ``(T, H, W, 3)`` uint8 frames."""

        import imageio.v3 as iio

        frames = iio.imread(self.video_bytes(chunk, index), extension=".mp4")
        return np.asarray(frames)[..., :3]


# ---------------------------------------------------------------------------
# Writing PhyWorld-format files (used by the simulator)
# ---------------------------------------------------------------------------


def encode_mp4(frames: np.ndarray, fps: int = 5) -> np.ndarray:
    """Encode frames to an MP4 byte stream like PhyWorld (needs imageio-ffmpeg)."""

    import imageio.v3 as iio

    with BytesIO() as buffer:
        iio.imwrite(buffer, frames, extension=".mp4", fps=fps)
        return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def write_phyworld_file(
    path: str | Path,
    positions: np.ndarray,
    init: np.ndarray,
    *,
    chunk_size: int = 1000,
    with_video: bool = False,
    video_size: int = 256,
) -> None:
    """Write videos in PhyWorld's merged layout (groups of chunk datasets)."""

    h5py = _h5py()
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        groups = {key: handle.create_group(key) for key in (POSITIONS, INIT)}
        if with_video:
            groups[VIDEOS] = handle.create_group(VIDEOS)
        for chunk_id, start in enumerate(range(0, len(positions), chunk_size)):
            stop = min(start + chunk_size, len(positions))
            name = f"{chunk_id:05d}"
            groups[POSITIONS].create_dataset(name, data=positions[start:stop])
            groups[INIT].create_dataset(name, data=init[start:stop])
            if with_video:
                dataset = groups[VIDEOS].create_dataset(
                    name, shape=(stop - start,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
                )
                for offset, index in enumerate(range(start, stop)):
                    frames = render_frames(positions[index], init[index, :2], video_size)
                    dataset[offset] = encode_mp4(frames)


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------


def describe(path: str | Path, *, max_items: int = 6) -> str:
    """Human-readable tree of an HDF5 file (first few items per group)."""

    h5py = _h5py()
    lines: list[str] = []

    def visit(name: str, node: Any, depth: int) -> None:
        indent = "  " * depth
        if hasattr(node, "keys"):
            keys = sorted(node.keys())
            lines.append(f"{indent}{name or '/'}/  ({len(keys)} items)")
            for key in keys[:max_items]:
                visit(key, node[key], depth + 1)
            if len(keys) > max_items:
                lines.append(f"{indent}  ... {len(keys) - max_items} more")
        else:
            lines.append(f"{indent}{name}  shape={node.shape} dtype={node.dtype}")

    with h5py.File(path, "r") as handle:
        visit("", handle, 0)
        for key, value in handle.attrs.items():
            lines.append(f"attr {key} = {value!r}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the layout of a PhyWorld HDF5 file")
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-items", type=int, default=6)
    arguments = parser.parse_args(argv)
    print(describe(arguments.path, max_items=arguments.max_items))
    with PhyWorldCollisionFile(arguments.path) as source:
        print(f"\ncollections: {source.collections or ['/']}")
        print(f"videos: {source.num_videos}")
        chunk = next(source.chunks())
        print(f"first chunk: {chunk.collection}/{chunk.name} ({len(chunk)} videos), "
              f"frames per video: {chunk.positions.shape[1]} (expected {NUM_FRAMES})")
        print(f"init (r1, r2, v1, v2) of first 3 videos:\n{chunk.init[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
