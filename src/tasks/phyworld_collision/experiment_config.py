"""NEO settings for the PhyWorld collision task.

Defaults follow ImageEditing (the paper's continuous-image task) with three
changes for physics:

* ``max_transition_length = 4`` to match the training deltas 1-4, so a
  program of length ``delta`` can be "one step of physics" repeated;
* a small codebook (``action_codebook_size = 8``) so the model cannot keep a
  separate code for every situation and has to reuse primitives;
* one GPU (``world_size`` is whatever ``torchrun`` gives; batch 128).

Every field can be overridden from the command line with ``--set name=value``.
The observation fields (``modality``, ``state_dim``, image size, ...) are
taken from the pretrained observation checkpoint, not set here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """One resolved NEO training condition (satisfies ``NEOTrainingConfig``)."""

    # schedule
    epochs: int = 50
    batch_size: int = 128
    num_workers: int = 4

    # latent program shape (state_dim / num_state_tokens come from the VAE)
    max_transition_length: int = 4
    state_dim: int = 256
    action_dim: int = 16
    num_state_tokens: int = 1
    num_action_tokens: int = 1
    policy_hidden_dim: int = 256
    transition_hidden_dim: int = 256
    policy_film_layers: int = 4
    transition_film_layers: int = 4
    dropout: float = 0.0

    # action codebook
    action_codebook_size: int = 8
    action_commitment_weight: float = 0.25
    action_entropy_weight: float = 0.1
    action_entropy_temperature: float = 1.0
    action_ema_decay: float = 0.99
    action_ema_epsilon: float = 1e-5
    action_tau_start: float = 0.3  # protocol fields; the EMA quantizer does not anneal
    action_tau_end: float = 0.1

    # optimisation
    learning_rate: float = 1e-3
    policy_learning_rate_scale: float = 0.25
    transition_learning_rate_scale: float = 0.5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    minimum_learning_rate_ratio: float = 0.1
    gradient_clip_norm: float = 1.0

    # objective
    reconstruction_weight: float = 1.0
    grounding_weight: float = 0.1
    state_prediction_weight: float = 0.5
    action_vq_weight: float = 0.5
    auto_reconstruction_weight: float = 0.0
    adaptive_grounding: bool = True
    foreground_weight: float = 10.0

    # MDL length control (>1 penalises longer programs)
    length_control_start: float = 1.01
    length_control_end: float = 1.01

    # frozen observation model
    freeze_observation_encoder: bool = True
    freeze_observation_decoder: bool = True

    # checkpoints
    checkpoint_interval_epochs: int = 5
    log_interval_steps: int = 50

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.max_transition_length < 1:
            raise ValueError("max_transition_length must be positive")
        if self.action_codebook_size < 2:
            raise ValueError("action_codebook_size must be at least 2")

    @property
    def action_tau_steps(self) -> int:
        return 0

    @property
    def length_control_coefficient(self) -> float:
        return self.length_control_start

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TrainingConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})


@dataclass(frozen=True, slots=True)
class Experiment:
    """Satisfies ``models.config.NEOExperimentConfig``."""

    training: TrainingConfig

    @property
    def length_control_coefficient(self) -> float:
        return self.training.length_control_start

    def scheduled_length_control(self, progress: float) -> float:
        """Linear schedule from ``length_control_start`` to ``length_control_end``."""

        progress = min(max(progress, 0.0), 1.0)
        start, end = self.training.length_control_start, self.training.length_control_end
        return start + (end - start) * progress


def _coerce(field_type: Any, text: str) -> Any:
    kind = field_type if isinstance(field_type, str) else getattr(field_type, "__name__", str(field_type))
    if kind == "bool":
        if text.lower() in ("1", "true", "yes", "on"):
            return True
        if text.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a boolean: {text!r}")
    if kind == "int":
        return int(text)
    if kind == "float":
        return float(text)
    return text


def apply_overrides(config: TrainingConfig, assignments: Sequence[str]) -> TrainingConfig:
    """Apply ``name=value`` strings, e.g. ``["epochs=10", "action_codebook_size=4"]``."""

    types = {f.name: f.type for f in fields(TrainingConfig)}
    updates: dict[str, Any] = {}
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        name = name.strip()
        if not separator or name not in types:
            raise ValueError(f"unknown setting {assignment!r}; valid names: {', '.join(sorted(types))}")
        updates[name] = _coerce(types[name], value.strip())
    return replace(config, **updates)


__all__ = ["Experiment", "TrainingConfig", "apply_overrides"]
