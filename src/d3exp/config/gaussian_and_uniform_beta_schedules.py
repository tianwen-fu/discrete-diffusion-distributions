from copy import deepcopy

import numpy as np

from .defaults import gaussian_cosine_default, gaussian_default, uniform_default

__all__ = ["beta_schedules"]

beta_schedules = {}
for transition in ("gaussian", "uniform"):
    for beta_end in (1e-3, 0.01, 0.05, 0.1, 0.5, 1.0):
        if transition == "gaussian":
            config_copy = deepcopy(gaussian_default)
        else:
            config_copy = deepcopy(uniform_default)
        assert config_copy.beta_schedule_type == "linear"
        config_copy.beta_schedule_kwargs["beta_end"] = beta_end
        beta_schedules[f"{transition}_linear_{beta_end}"] = config_copy
        step_config_copy = deepcopy(config_copy)
        step_config_copy.beta_schedule_type = "step"
        step_config_copy.beta_schedule_kwargs = dict(
            step_values=np.linspace(0.02, beta_end, 5).tolist()
        )
        beta_schedules[f"{transition}_step_{beta_end}"] = step_config_copy

    beta_schedules[f"{transition}_cosine"] = gaussian_cosine_default
    if transition == "uniform":
        beta_schedules[f"{transition}_cosine"].transition_type = "uniform"
    beta_schedules[f"{transition}_jsd"] = deepcopy(gaussian_cosine_default)
    beta_schedules[f"{transition}_jsd"].beta_schedule_type = "jsd"
