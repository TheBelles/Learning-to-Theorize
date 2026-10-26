"""PyTorch datasets over the episode files written by :mod:`.convert`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from tasks.phyworld_collision.data.episodes import read_attributes


MODALITIES = ("pixels", "states")


def _h5py() -> Any:
    import h5py

    return h5py


class _LazyHDF5:
    """Open the HDF5 file on first use in each process (h5py handles cannot be forked)."""

    def __init__(self, path: str | Path, modality: str) -> None:
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
        self.path = Path(path).expanduser().resolve(strict=True)
        self.modality = modality
        self.attributes = read_attributes(self.path)
        if modality == "pixels" and not bool(self.attributes["has_pixels"]):
            raise ValueError(f"{self.path} was converted with --states-only")
        self.key = "grids" if modality == "pixels" else "states"
        self.num_episodes = int(self.attributes["dataset_length"])
        self._handle: Any | None = None

    @property
    def handle(self) -> Any:
        if self._handle is None:
            self._handle = _h5py().File(self.path, "r")
        return self._handle

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state


class EpisodeDataset(Dataset):
    """Episodes as ``(4, S, S, 6)`` uint8 (pixels) or ``(4, 10)`` float32 (states).

    Items are ``(observations, episode_index)`` so evaluation can look up the
    metadata (delta, collisions, true positions) of each episode.
    """

    def __init__(self, path: str | Path, modality: str) -> None:
        self._file = _LazyHDF5(path, modality)

    @property
    def attributes(self) -> dict:
        return self._file.attributes

    @property
    def path(self) -> Path:
        return self._file.path

    def __len__(self) -> int:
        return self._file.num_episodes

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        array = np.asarray(self._file.handle[self._file.key][index])
        return torch.from_numpy(array), index


class ObservationDataset(Dataset):
    """Every single observation of an episode file, for observation-model pretraining."""

    def __init__(self, path: str | Path, modality: str) -> None:
        self._file = _LazyHDF5(path, modality)

    @property
    def attributes(self) -> dict:
        return self._file.attributes

    @property
    def path(self) -> Path:
        return self._file.path

    def __len__(self) -> int:
        return 4 * self._file.num_episodes

    def __getitem__(self, index: int) -> torch.Tensor:
        episode, slot = divmod(index, 4)
        return torch.from_numpy(np.asarray(self._file.handle[self._file.key][episode, slot]))


@dataclass(frozen=True, slots=True)
class EpisodeBatch:
    """A collated batch: ``observations`` is ``(B, 4, ...)``."""

    observations: torch.Tensor
    indices: torch.Tensor

    def to(self, device: torch.device | str) -> "EpisodeBatch":
        return EpisodeBatch(self.observations.to(device, non_blocking=True), self.indices)

    def __len__(self) -> int:
        return self.observations.shape[0]


def collate_episodes(items: Sequence[tuple[torch.Tensor, int]]) -> EpisodeBatch:
    return EpisodeBatch(
        observations=torch.stack([item[0] for item in items]),
        indices=torch.tensor([item[1] for item in items], dtype=torch.long),
    )


@dataclass(frozen=True, slots=True)
class EpisodeMetadata:
    """Per-episode ground truth used by the evaluation."""

    delta: np.ndarray  # (N,)
    collides: np.ndarray  # (N, 2) support, query
    positions: np.ndarray  # (N, 4, 2, 2, 2)
    radii: np.ndarray  # (N, 2, 2)
    init: np.ndarray  # (N, 2, 4)


def load_metadata(path: str | Path) -> EpisodeMetadata:
    with _h5py().File(path, "r") as handle:
        return EpisodeMetadata(
            delta=np.asarray(handle["delta"][:], dtype=np.int64),
            collides=np.asarray(handle["collides"][:], dtype=bool),
            positions=np.asarray(handle["positions"][:], dtype=np.float64),
            radii=np.asarray(handle["radii"][:], dtype=np.float64),
            init=np.asarray(handle["init"][:], dtype=np.float64),
        )


__all__ = [
    "EpisodeBatch",
    "EpisodeDataset",
    "EpisodeMetadata",
    "MODALITIES",
    "ObservationDataset",
    "collate_episodes",
    "load_metadata",
]
