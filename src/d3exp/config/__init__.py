from typing import Dict

from .base import Config
from .defaults import *

NAMED_CONFIGS = {
    "DEFAULTS": dict(
        uniform=uniform_default,
        gaussian=gaussian_default,
        gaussian_cosine=gaussian_cosine_default,
        absorbing=absorbing_default,
        random_double_stochastic=random_double_stochastic_default,
    ),
}


def get_configs(name: str) -> Dict[str, Config]:
    return NAMED_CONFIGS[name]
