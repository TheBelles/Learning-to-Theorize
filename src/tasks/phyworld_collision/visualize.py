"""Small PNG helpers for checking reconstructions and predictions by eye."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def _to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.max(initial=0.0) <= 1.0:
        image = image * 255.0
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def frame_pair(observation: np.ndarray) -> np.ndarray:
    """``(S, S, 6)`` observation -> ``(S, 2S, 3)``: earlier frame | later frame."""

    observation = _to_uint8(observation)
    return np.concatenate([observation[..., :3], observation[..., 3:6]], axis=1)


def save_rows(path: str | Path, rows: Sequence[Sequence[np.ndarray]], *, scale: int = 2, gap: int = 4) -> None:
    """Save a grid; each cell is an ``(S, S, 6)`` observation shown as two frames."""

    from PIL import Image

    cells = [[frame_pair(cell) for cell in row] for row in rows]
    height = max(cell.shape[0] for row in cells for cell in row)
    width = max(cell.shape[1] for row in cells for cell in row)
    columns = max(len(row) for row in cells)
    canvas = np.full(
        (len(cells) * (height + gap) + gap, columns * (width + gap) + gap, 3), 128, dtype=np.uint8
    )
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            y, x = gap + r * (height + gap), gap + c * (width + gap)
            canvas[y : y + cell.shape[0], x : x + cell.shape[1]] = cell
    image = Image.fromarray(canvas)
    if scale != 1:
        image = image.resize((canvas.shape[1] * scale, canvas.shape[0] * scale), Image.NEAREST)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


__all__ = ["frame_pair", "save_rows"]
