import abc
from typing import List

import jax.numpy as jnp

from d3exp.config import Config


class BetaSchedule(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_betas(self, num_steps: int) -> jnp.ndarray:
        pass


# Linear, Cosine, and JSD schedules are from the D3PM code
# https://github.com/google-research/google-research/tree/master/d3pm
# following the tradition of the D3PM code, we use float64


class LinearBetaSchedule(BetaSchedule):
    def __init__(self, beta_start: float, beta_end: float):
        self.beta_start = beta_start
        self.beta_end = beta_end

    def get_betas(self, num_steps: int) -> jnp.ndarray:
        return jnp.linspace(
            self.beta_start, self.beta_end, num_steps, dtype=jnp.float64
        )


class CosineBetaSchedule(BetaSchedule):
    def get_betas(self, num_steps: int) -> jnp.ndarray:
        steps = jnp.arange(num_steps + 1, dtype=jnp.float64) / num_steps
        alpha_bar = jnp.cos((steps + 0.008) / 1.008 * jnp.pi / 2)
        betas = jnp.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
        return betas


class JSDBetaSchedule(BetaSchedule):
    def get_betas(self, num_steps: int):
        return 1 / jnp.linspace(num_steps, 1, num_steps, dtype=jnp.float64)


class StepBetaSchedule(BetaSchedule):
    def __init__(self, step_values: List[float]):
        self.num_steps = len(step_values)
        self.step_values = jnp.array(step_values, dtype=jnp.float64)

    def get_betas(self, num_steps: int) -> jnp.ndarray:
        selected_steps = jnp.linspace(
            0, self.num_steps, endpoint=False, num=num_steps
        ).astype(jnp.int32)
        betas = self.step_values[selected_steps]
        return betas


BETA_SCHEDULE_CLASSES = dict(
    linear=LinearBetaSchedule,
    cosine=CosineBetaSchedule,
    jsd=JSDBetaSchedule,
    step=StepBetaSchedule,
)


def get_beta_schedule(config: Config):
    return BETA_SCHEDULE_CLASSES[config.beta_schedule_type](
        **config.beta_schedule_kwargs
    )
