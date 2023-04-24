from copy import deepcopy

from .defaults import defaults_dict

__all__ = ["over_peakiness"]

over_peakiness = {}
for num_modes in (1, 5):
    for tau in (0.01, 0.05, 0.1, 1):
        for name, config in defaults_dict.items():
            config = deepcopy(config)
            config.data_tau = tau
            config.n_modes = num_modes
            over_peakiness[f"{name}_tau_{tau}_M_{num_modes}"] = config
