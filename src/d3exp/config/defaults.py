from copy import deepcopy

from .base import Config

__all__ = (
    "uniform_default",
    "absorbing_default",
    "gaussian_default",
    "gaussian_cosine_default",
    "random_double_stochastic_default",
)

uniform_default = Config(
    work_dir="/dev/null",
    batch_size=512,
    data_shape=(),
    seed=0,
    epochs=5,
    log_every=20,
    grad_clip=1.0,
    warmup_steps=10,
    learning_rate=2e-4,
    sample_max_retries=100,
    num_classes=50,
    cls_embed_dim=64,
    time_embed_dim=64,
    x_features_in=[64, 64],
    x_features_out=[
        64,
    ],
    num_timesteps=1000,
    eps=1.0e-6,
    hybrid_coeff=0.001,
    transition_type="uniform",
    transition_kwargs=dict(transition_bands=None),
    beta_schedule_type="linear",
    beta_schedule_kwargs=dict(beta_start=0.02, beta_end=1.0),
    dataset_size=65536,
    n_modes=1,
    data_tau=0.01,
)

gaussian_default = deepcopy(uniform_default)
gaussian_default.update(
    dict(
        transition_type="gaussian",
        beta_schedule_type="linear",
        beta_schedule_kwargs=dict(beta_start=1e-4, beta_end=0.02),
    )
)

gaussian_cosine_default = deepcopy(gaussian_default)
gaussian_cosine_default.update(
    dict(
        beta_schedule_type="cosine",
        beta_schedule_kwargs=dict(),
    )
)

absorbing_default = deepcopy(uniform_default)
absorbing_default.update(
    dict(
        transition_type="absorbing",
        transition_kwargs=dict(stationary_state=0),  # TODO: some experiments with this
        beta_schedule_type="jsd",
        beta_schedule_kwargs=dict(),
    )
)

random_double_stochastic_default = deepcopy(uniform_default)
random_double_stochastic_default.update(
    dict(
        transition_type="random_double_stochastic",
        transition_kwargs=dict(beta_max=6),
        beta_schedule_type="linear",
        beta_schedule_kwargs=dict(beta_start=1.0, beta_end=5.5),
    )
)
