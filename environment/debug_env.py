import numpy as np
from helpers.environment import ActionSpace


class DebugEnv:
    def __init__(self):
        self.action_list = list(range(len(ActionSpace.action_name_list)))

    def step(self, _action):
        obs = {"pov": np.random.randint(0, 255, (64, 64, 3))}
        _reward = None
        _info = None
        return obs, _reward, False, _info

    def reset(self):
        obs, _, _, _ = self.step(0)
        return obs

    def close(self):
        return
