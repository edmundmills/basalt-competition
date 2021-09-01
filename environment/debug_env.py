import numpy as np


class DebugEnv:
    def __init__(self):
        self.action_list = list(range(100))

    def step(self, _action):
        obs = {"pov": np.random.randint(0, 255, (64, 64, 3)),
               "inventory": {'snowball': 1},
               "equipped_items": {"mainhand": {'type': 'snowball'}}}
        _reward = None
        _info = None
        return obs, _reward, False, _info

    def reset(self):
        obs, _, _, _ = self.step(0)
        return obs

    def close(self):
        return
