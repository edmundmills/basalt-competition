import contexts.minerl.environment as minerl_env
from contexts.minerl.environment import Context, MineRLContext

from core.state import State

from collections import deque

import gym
from omegaconf import OmegaConf
import torch as th


def start_env(config: OmegaConf, debug_env: bool = False) -> gym.Env:
    """Looks up the env from config and starts the environment."""
    context = config.context.name
    if context == 'MineRL':
        env = minerl_env.start_env(config, debug_env)
    return env


def create_context(config: OmegaConf) -> Context:
    """Looks up the context from config and returns the correct context."""
    if config.context.name == 'MineRL':
        context = MineRLContext(config)
    return context
