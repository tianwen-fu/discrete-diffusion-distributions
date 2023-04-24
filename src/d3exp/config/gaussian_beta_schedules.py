from copy import deepcopy

import numpy as np

from .defaults import gaussian_cosine_default, gaussian_default

__all__ = ["gaussian_beta_schedules"]

gaussian_beta_schedules = {}
for beta_end in (1e-3, 0.01, 0.05, 0.1, 0.5, 1.0):
    config_copy = deepcopy(gaussian_default)
    assert config_copy.beta_schedule_type == "linear"
    config_copy.beta_schedule_kwargs["beta_end"] = beta_end
    gaussian_beta_schedules[f"linear_{beta_end}"] = config_copy
    step_config_copy = deepcopy(config_copy)
    step_config_copy.beta_schedule_type = "step"
    step_config_copy.beta_schedule_kwargs = dict(
        step_values=np.linspace(1e-4, beta_end, 5).tolist()
    )
    gaussian_beta_schedules[f"step_{beta_end}"] = step_config_copy


gaussian_beta_schedules["cosine"] = gaussian_cosine_default
gaussian_beta_schedules["jsd"] = deepcopy(gaussian_cosine_default)
gaussian_beta_schedules["jsd"].beta_schedule_type = "jsd"
