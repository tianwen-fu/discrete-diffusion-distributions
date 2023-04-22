from .base import Config
from .uniform import *

NAMED_CONFIGS = {
    "UNIFORM_DEFAULT": uniform_default,
}


def get_config(name: str) -> Config:
    return NAMED_CONFIGS[name]
