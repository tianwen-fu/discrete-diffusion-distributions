from copy import deepcopy

from .defaults import random_double_stochastic_default

__all__ = ["doubly_stochastic_beta_schedules"]

doubly_stochastic_beta_schedules = {}

for linear_beta_max in range(3, 9):
    config = deepcopy(random_double_stochastic_default)
    config.transition_kwargs["beta_max"] = linear_beta_max
    config.beta_schedule_kwargs["beta_end"] = linear_beta_max - 0.5
    doubly_stochastic_beta_schedules[f"linear_bmax_{linear_beta_max}"] = config

cosine_config = deepcopy(random_double_stochastic_default)
cosine_config.beta_schedule_type = "cosine"
cosine_config.beta_schedule_kwargs = dict()

jsd_config = deepcopy(random_double_stochastic_default)
jsd_config.beta_schedule_type = "jsd"
jsd_config.beta_schedule_kwargs = dict()
