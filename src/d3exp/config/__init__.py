from typing import Dict

from .absorbing_center import *
from .base import Config
from .defaults import *
from .doubly_stochastic_beta_schedules import *
from .gaussian_and_uniform_beta_schedules import *
from .over_peakiness import *

NAMED_CONFIGS = {
    "DEFAULTS": dict(
        uniform=uniform_default,
        gaussian=gaussian_default,
        gaussian_cosine=gaussian_cosine_default,
        absorbing=absorbing_default,
        random_double_stochastic=random_double_stochastic_default,
    ),
    "OVER_PEAKINESS": over_peakiness,
    "DOUBLY_STOCHASTIC_BETAS": doubly_stochastic_beta_schedules,
    "ABSORBING_CENTER_M1": absorbing_center_M1,
    "ABSORBING_CENTER_M5": absorbing_center_M5,
    "BETA_SCHEDULES": beta_schedules,
}


def get_configs(name: str) -> Dict[str, Config]:
    return NAMED_CONFIGS[name]
