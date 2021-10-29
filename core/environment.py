import contexts.minerl.environment as minerl_env
from core.state import State

import gym

from collections import deque
import torch as th


def start_env(config, debug_env=False):
    context = config.context.name
    if context == 'MineRL':
        env = minerl_env.start_env(config, debug_env)
    return env


def create_context(config):
    if config.context.name == 'MineRL':
        context = minerl_env.MineRLContext(config)
    return context
