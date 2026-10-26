"""Observation models: a conv VAE for stacked frames and an MLP VAE for state vectors.

Both encoders return ``(posterior.mean, posterior)`` with ``mean`` shaped
``(B, num_state_tokens, state_dim)``, which is the contract of the shared
:class:`models.neo.NEO`.  The conv VAE reuses the residual blocks of the
ImageEditing VAE and generalises its sizes: ``image_size`` 64 with an 8x8
bottleneck gives three stride-2 stages, as 32 -> 4 does for ImageEditing.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from tasks.image_editing.models.vae import GaussianPosterior, ResidualBlock, ResidualUpBlock
from tasks.phyworld_collision.physics import STATE_DIM


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    """Architecture of the observation model; saved inside every checkpoint."""

    modality: str = "pixels"  # "pixels" or "states"
    state_dim: int = 256
    num_state_tokens: int = 1
    dropout: float = 0.0
    # pixels
    image_size: int = 64
    channels: int = 6
    base_channels: int = 64
    bottleneck_channels: int = 256
    bottleneck_size: int = 8
    # states
    input_dim: int = STATE_DIM
    hidden_dim: int = 256

    def __post_init__(self) -> None:
        if self.modality not in ("pixels", "states"):
            raise ValueError(f"modality must be 'pixels' or 'states', got {self.modality!r}")
        if self.state_dim < 1 or self.num_state_tokens < 1:
            raise ValueError("state_dim and num_state_tokens must be positive")
        if self.modality == "pixels":
            ratio = self.image_size / self.bottleneck_size
            if ratio < 1 or not float(ratio).is_integer() or not math.log2(ratio).is_integer():
                raise ValueError("image_size / bottleneck_size must be a power of two")

    @property
    def latent_dim(self) -> int:
        return self.state_dim * self.num_state_tokens

    @property
    def num_downsamples(self) -> int:
        return int(math.log2(self.image_size // self.bottleneck_size))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ObservationConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})


def _stage_channels(config: ObservationConfig) -> list[int]:
    channels = [config.base_channels * 2 ** (i + 1) for i in range(config.num_downsamples)]
    if channels:
        channels[-1] = config.bottleneck_channels
    return channels


def to_float_channels_first(image: Tensor, channels: int, image_size: int) -> Tensor:
    """``(B, H, W, C)`` uint8 or ``[0, 1]`` floats -> ``(B, C, H, W)`` float32."""

    if image.ndim != 4:
        raise ValueError(f"image must be 4-dimensional, got {tuple(image.shape)}")
    if image.shape[-1] == channels and image.shape[1] == image_size:
        image = image.permute(0, 3, 1, 2)
    if tuple(image.shape[1:]) != (channels, image_size, image_size):
        raise ValueError(
            f"expected (B, {image_size}, {image_size}, {channels}) images, got {tuple(image.shape)}"
        )
    if image.dtype == torch.uint8:
        return image.float() / 255.0
    return image.float()


class ConvEncoder(nn.Module):
    def __init__(self, config: ObservationConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(config.channels, config.base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(config.base_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(config.base_channels, config.base_channels, 1, config.dropout),
        )
        stages = []
        previous = config.base_channels
        for out_channels in _stage_channels(config):
            stages.append(ResidualBlock(previous, out_channels, 2, config.dropout))
            previous = out_channels
        self.stages = nn.Sequential(*stages)
        flattened = config.bottleneck_channels * config.bottleneck_size**2
        self.to_mean = nn.Linear(flattened, config.latent_dim)
        self.to_logvar = nn.Linear(flattened, config.latent_dim)

    def forward(self, image: Tensor) -> tuple[Tensor, GaussianPosterior]:
        config = self.config
        x = to_float_channels_first(image, config.channels, config.image_size)
        features = self.stages(self.stem(x)).flatten(1)
        shape = (x.shape[0], config.num_state_tokens, config.state_dim)
        posterior = GaussianPosterior(
            mean=self.to_mean(features).view(shape),
            log_variance=self.to_logvar(features).view(shape).clamp(-20.0, 10.0),
        )
        return posterior.mean, posterior


class ConvDecoder(nn.Module):
    def __init__(self, config: ObservationConfig) -> None:
        super().__init__()
        self.config = config
        flattened = config.bottleneck_channels * config.bottleneck_size**2
        self.from_state = nn.Linear(config.latent_dim, flattened)
        encoder_channels = [config.base_channels, *_stage_channels(config)]
        blocks = []
        previous = config.bottleneck_channels
        for out_channels in reversed(encoder_channels[:-1]):
            blocks.append(ResidualUpBlock(previous, out_channels, config.dropout))
            previous = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Conv2d(previous, config.channels, 3, padding=1), nn.Sigmoid())

    def forward(self, state: Tensor) -> Tensor:
        config = self.config
        expected = (config.num_state_tokens, config.state_dim)
        if state.ndim != 3 or tuple(state.shape[1:]) != expected:
            raise ValueError(f"state must be (B, {expected[0]}, {expected[1]}), got {tuple(state.shape)}")
        x = self.from_state(state.reshape(state.shape[0], -1)).view(
            state.shape[0], config.bottleneck_channels, config.bottleneck_size, config.bottleneck_size
        )
        return self.head(self.blocks(x)).permute(0, 2, 3, 1)


class StateEncoder(nn.Module):
    def __init__(self, config: ObservationConfig) -> None:
        super().__init__()
        self.config = config
        self.body = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.to_mean = nn.Linear(config.hidden_dim, config.latent_dim)
        self.to_logvar = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, state_vector: Tensor) -> tuple[Tensor, GaussianPosterior]:
        config = self.config
        if state_vector.ndim != 2 or state_vector.shape[1] != config.input_dim:
            raise ValueError(f"expected (B, {config.input_dim}) states, got {tuple(state_vector.shape)}")
        hidden = self.body(state_vector.float())
        shape = (state_vector.shape[0], config.num_state_tokens, config.state_dim)
        posterior = GaussianPosterior(
            mean=self.to_mean(hidden).view(shape),
            log_variance=self.to_logvar(hidden).view(shape).clamp(-20.0, 10.0),
        )
        return posterior.mean, posterior


class StateDecoder(nn.Module):
    def __init__(self, config: ObservationConfig) -> None:
        super().__init__()
        self.config = config
        self.body = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.body(state.reshape(state.shape[0], -1))


class ObservationVAE(nn.Module):
    """Encoder/decoder pair with ``encoder.*`` / ``decoder.*`` parameter names."""

    def __init__(self, config: ObservationConfig) -> None:
        super().__init__()
        self.config = config
        if config.modality == "pixels":
            self.encoder: nn.Module = ConvEncoder(config)
            self.decoder: nn.Module = ConvDecoder(config)
        else:
            self.encoder = StateEncoder(config)
            self.decoder = StateDecoder(config)

    def forward(self, observation: Tensor, *, sample: bool = True) -> tuple[Tensor, GaussianPosterior]:
        mean, posterior = self.encoder(observation)
        state = mean
        if sample and self.training:
            noise = torch.randn_like(mean)
            state = mean + noise * torch.exp(0.5 * posterior.log_variance)
        return self.decoder(state), posterior


__all__ = [
    "ConvDecoder",
    "ConvEncoder",
    "ObservationConfig",
    "ObservationVAE",
    "StateDecoder",
    "StateEncoder",
    "to_float_channels_first",
]
