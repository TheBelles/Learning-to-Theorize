"""Compose the shared NEO model for the PhyWorld collision task.

The theory programmer, program executor and EMA action quantizer are the
ImageEditing modules (four FiLM blocks each); the observation model and the
objective are this task's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.neo import NEO
from tasks.image_editing.models.program_execution import ProgramExecutor
from tasks.image_editing.models.quantizer import ActionQuantizer, ActionQuantizerConfig
from tasks.image_editing.models.theory_programmer import LatentProgramConfig, TheoryProgrammer
from tasks.phyworld_collision.experiment_config import Experiment, TrainingConfig
from tasks.phyworld_collision.models.vae import ObservationConfig, ObservationVAE
from tasks.phyworld_collision.objective import CollisionMetrics, PixelObjective, StateObjective


def _latent_program_config(training: TrainingConfig, hidden: int, film_layers: int) -> LatentProgramConfig:
    return LatentProgramConfig(
        state_dim=training.state_dim,
        action_dim=training.action_dim,
        num_state_tokens=training.num_state_tokens,
        num_action_tokens=training.num_action_tokens,
        hidden_dim=hidden,
        num_film_layers=film_layers,
        dropout=training.dropout,
        max_transition_length=training.max_transition_length,
    )


def build_objective(observation: ObservationConfig, training: TrainingConfig):
    if observation.modality == "pixels":
        return PixelObjective(observation.image_size, observation.channels, training.foreground_weight)
    return StateObjective(observation.input_dim)


class CollisionNEO(NEO[CollisionMetrics]):
    """Shared NEO with this task's observation model and objective."""

    def __init__(self, experiment: Experiment, observation: ObservationConfig) -> None:
        training = experiment.training
        if (training.state_dim, training.num_state_tokens) != (
            observation.state_dim,
            observation.num_state_tokens,
        ):
            raise ValueError("training.state_dim/num_state_tokens must match the observation model")
        vae = ObservationVAE(observation)
        super().__init__(
            experiment,
            encoder=vae.encoder,
            decoder=vae.decoder,
            objective=build_objective(observation, training),
            theory_programmer=TheoryProgrammer(
                _latent_program_config(training, training.policy_hidden_dim, training.policy_film_layers)
            ),
            program_executor=ProgramExecutor(
                _latent_program_config(
                    training, training.transition_hidden_dim, training.transition_film_layers
                )
            ),
            quantizer=ActionQuantizer(
                ActionQuantizerConfig(
                    codebook_size=training.action_codebook_size,
                    action_dim=training.action_dim,
                    commitment_weight=training.action_commitment_weight,
                    entropy_weight=training.action_entropy_weight,
                    entropy_temperature=training.action_entropy_temperature,
                    ema_decay=training.action_ema_decay,
                    ema_epsilon=training.action_ema_epsilon,
                )
            ),
        )
        self.observation_config = observation


def build_neo(experiment: Experiment, observation: ObservationConfig) -> CollisionNEO:
    return CollisionNEO(experiment, observation)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def load_observation_checkpoint(path: str | Path) -> tuple[ObservationConfig, dict[str, torch.Tensor]]:
    """Read a checkpoint written by :mod:`.observation_pretraining`."""

    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    if "observation_config" not in payload:
        raise ValueError(f"{path} has no observation_config; was it written by observation_pretraining?")
    config = ObservationConfig.from_dict(payload["observation_config"])
    state = {
        key: value
        for key, value in payload["model_state_dict"].items()
        if key.startswith(("encoder.", "decoder."))
    }
    if not state:
        raise ValueError(f"no encoder/decoder tensors in {path}")
    return config, state


def load_observation_weights(model: nn.Module, state: dict[str, torch.Tensor]) -> int:
    """Copy pretrained ``encoder.*`` / ``decoder.*`` tensors into a NEO model."""

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"unexpected observation keys: {unexpected[:5]}")
    still_missing = [k for k in missing if k.startswith(("encoder.", "decoder."))]
    if still_missing:
        raise ValueError(f"observation checkpoint is missing keys: {still_missing[:5]}")
    return len(state)


def load_neo_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> CollisionNEO:
    """Rebuild a trained NEO from a checkpoint written by :mod:`.theorizer_runner`."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    training = TrainingConfig.from_dict(payload["training_config"])
    observation = ObservationConfig.from_dict(payload["observation_config"])
    model = build_neo(Experiment(training), observation)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


__all__ = [
    "CollisionNEO",
    "build_neo",
    "build_objective",
    "load_neo_checkpoint",
    "load_observation_checkpoint",
    "load_observation_weights",
]
