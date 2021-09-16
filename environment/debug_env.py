from helpers.environment import ActionSpace, ObservationSpace

import numpy as np


class DebugEnv:
    def __init__(self):
        self.action_list = ActionSpace.actions()
        self.inventory = ObservationSpace.starting_inventory()

    def step(self, _action):
        obs = {"pov": np.random.randint(0, 255, (64, 64, 3)),
               "inventory": self.inventory,
               "equipped_items": {"mainhand": {'type': 'snowball'}}}
        _reward = None
        _info = None
        done = np.random.choice([True, False], p=[.02, .98])
        return obs, _reward, done, _info

    def reset(self):
        obs, _, _, _ = self.step(0)
        return obs

    def close(self):
        return
