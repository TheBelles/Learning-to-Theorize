"""Observation semantics supplied to the shared NEO model.

Pixels
    Observations are ``(64, 64, 6)`` uint8 (two stacked RGB frames).  The
    loss is a **foreground-weighted L1**: pixels that belong to a ball in the
    target get ``foreground_weight`` times the weight of white background.
    Without it, a model can reach a low plain L1 by drawing an empty white
    image, because the balls cover only a few percent of the pixels.
    Grounding snaps decoded images to 8 bits, as in ImageEditing; there is no
    exact match for continuous images.

States
    Observations are the 10-number vectors of :func:`physics.state_vector`;
    the loss is MSE and grounding is the identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


QUANTISATION_LEVELS = 255
FOREGROUND_THRESHOLD = 0.9  # a pixel is ball if any channel is darker than this


@dataclass(frozen=True, slots=True)
class CollisionMetrics:
    """Mean absolute error overall and on ball pixels (pixels), or MSE (states)."""

    l1: float
    foreground_l1: float


def _split_pairs(batch: Tensor) -> tuple[Tensor, Tensor]:
    trailing = batch.shape[2:]
    return batch[:, ::2].reshape(-1, *trailing), batch[:, 1::2].reshape(-1, *trailing)


def _validate_pairs(batch: Tensor, *, is_eval: bool, trailing: tuple[int, ...]) -> None:
    if batch.ndim != 2 + len(trailing):
        raise ValueError(f"episodes must be (B, pairs*2, {trailing}), got {tuple(batch.shape)}")
    if batch.shape[1] < 2 or batch.shape[1] % 2:
        raise ValueError("episodes must contain an even, non-zero number of observations")
    if tuple(batch.shape[2:]) != trailing:
        raise ValueError(f"observations must be {trailing}, got {tuple(batch.shape[2:])}")
    if is_eval and batch.shape[1] != 4:
        raise ValueError("evaluation needs one support and one query pair per episode")


def foreground_mask(target: Tensor) -> Tensor:
    """``(..., H, W, C)`` in [0, 1] -> ``(..., H, W, 1)`` mask of non-white pixels per frame.

    The six channels are two RGB frames; each frame gets its own mask.
    """

    frames = target.unflatten(-1, (-1, 3))  # (..., H, W, F, 3)
    mask = (frames.min(dim=-1).values < FOREGROUND_THRESHOLD).float()  # (..., H, W, F)
    return mask.repeat_interleave(3, dim=-1)


class PixelObjective:
    """Pairing, weighted L1, 8-bit grounding and metrics for stacked frames."""

    def __init__(self, image_size: int, channels: int, foreground_weight: float) -> None:
        self.shape = (image_size, image_size, channels)
        self.foreground_weight = float(foreground_weight)

    def validate_batch(self, batch: Tensor, *, is_eval: bool) -> None:
        _validate_pairs(batch, is_eval=is_eval, trailing=self.shape)

    def split_batch(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        return _split_pairs(batch)

    @staticmethod
    def normalize(target: Tensor) -> Tensor:
        return target.float() / 255.0 if target.dtype == torch.uint8 else target.float()

    def weights(self, target: Tensor) -> Tensor:
        return 1.0 + (self.foreground_weight - 1.0) * foreground_mask(target)

    def per_sample_reconstruction_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        target = self.normalize(target)
        weights = self.weights(target)
        error = (prediction.float() - target).abs() * weights
        return error.flatten(1).sum(dim=1) / weights.flatten(1).sum(dim=1)

    def reconstruction_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.per_sample_reconstruction_loss(prediction, target).mean()

    def decode_prediction(self, prediction: Tensor) -> Tensor:
        return torch.round(prediction.clamp(0.0, 1.0) * QUANTISATION_LEVELS) / QUANTISATION_LEVELS

    def exact_match(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.zeros(target.shape[0], dtype=torch.bool, device=prediction.device)

    @torch.no_grad()
    def metrics(self, prediction: Tensor, target: Tensor) -> CollisionMetrics:
        target = self.normalize(target)
        error = (prediction.float() - target).abs()
        mask = torch.maximum(foreground_mask(target), foreground_mask(prediction.float()))
        foreground = (error * mask).sum() / mask.sum().clamp_min(1.0)
        return CollisionMetrics(l1=error.mean().item(), foreground_l1=foreground.item())

    def empty_metrics(self) -> CollisionMetrics:
        return CollisionMetrics(l1=0.0, foreground_l1=0.0)


class StateObjective:
    """Pairing, MSE and metrics for state vectors."""

    def __init__(self, input_dim: int) -> None:
        self.shape = (input_dim,)

    def validate_batch(self, batch: Tensor, *, is_eval: bool) -> None:
        _validate_pairs(batch, is_eval=is_eval, trailing=self.shape)

    def split_batch(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        inputs, targets = _split_pairs(batch)
        return inputs.float(), targets.float()

    def per_sample_reconstruction_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        return (prediction.float() - target.float()).pow(2).mean(dim=1)

    def reconstruction_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.per_sample_reconstruction_loss(prediction, target).mean()

    def decode_prediction(self, prediction: Tensor) -> Tensor:
        return prediction.float()

    def exact_match(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.zeros(target.shape[0], dtype=torch.bool, device=prediction.device)

    @torch.no_grad()
    def metrics(self, prediction: Tensor, target: Tensor) -> CollisionMetrics:
        error = (prediction.float() - target.float()).abs()
        return CollisionMetrics(l1=error.mean().item(), foreground_l1=error.mean().item())

    def empty_metrics(self) -> CollisionMetrics:
        return CollisionMetrics(l1=0.0, foreground_l1=0.0)


__all__ = [
    "CollisionMetrics",
    "FOREGROUND_THRESHOLD",
    "PixelObjective",
    "StateObjective",
    "foreground_mask",
]
