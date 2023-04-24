from copy import deepcopy

from .defaults import random_double_stochastic_default

__all__ = ["doubly_stochastic_beta_schedules"]

doubly_stochastic_beta_schedules = {}

for linear_beta_max in range(2, 9):
    config = deepcopy(random_double_stochastic_default)
    config.transition_kwargs["beta_max"] = linear_beta_max
    config.beta_schedule_kwargs["step_values"] = list(range(1, linear_beta_max))
    doubly_stochastic_beta_schedules[f"linear_bmax_{linear_beta_max}"] = config
