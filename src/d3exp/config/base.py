from dataclasses import dataclass
from typing import Any, Dict, Sequence


@dataclass
class Config:
    # training
    work_dir: str
    batch_size: int
    data_shape: tuple
    seed: int
    epochs: int
    log_every: int
    grad_clip: int
    warmup_steps: int
    learning_rate: float
    sample_max_retries: int

    # model
    num_classes: int
    cls_embed_dim: int
    time_embed_dim: int
    x_features_in: Sequence[int]  # MLPs before adding time embedding
    x_features_out: Sequence[int]  # MLPs after adding time embedding

    # diffusion
    num_timesteps: int
    eps: float
    hybrid_coeff: float
    transition_type: str
    transition_kwargs: Dict[str, Any]
    beta_schedule_type: str
    beta_schedule_kwargs: Dict[str, Any]

    # data
    dataset_size: int
    n_modes: int
    data_tau: float

    def update(self, d: Dict[str, Any]) -> None:
        for k, v in d.items():
            setattr(self, k, v)
