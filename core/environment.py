import contexts.minerl.environment as minerl_env
from core.state import State

import gym

from collections import deque
import torch as th


def start_env(config, debug_env=False):
    context = config.context.name
    if context == 'MineRL':
        env = minerl_env.start_env(config, debug_env)
    if config.n_observation_frames > 1:
        env = FrameStack(env)
    return env


class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.n_observation_frames = config.n_observation_frames
        self.framestack = deque(maxlen=self.n_observation_frames)

    def observation(state):
        pov = state.spatial
        while len(self.framestack) < self.n_observation_frames:
            self.framestack.append(pov)
        self.framestack.append(pov)
        spatial = th.cat(list(self.framestack), dim=0)
        state = list(state)
        state[0] = spatial
        state = State(*state)
        return state
