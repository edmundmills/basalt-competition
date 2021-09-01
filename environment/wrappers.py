from helpers.environment import ObservationSpace

import gym
import numpy as np
import random


class ActionShaping(gym.ActionWrapper):
    """
    The default MineRL action space is the following dict:

    Dict(attack:Discrete(2),
         back:Discrete(2),
         camera:Box(low=-180.0, high=180.0, shape=(2,)),
         craft:Enum(crafting_table,none,planks,stick,torch),
         equip:Enum(air,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe),
         forward:Discrete(2),
         jump:Discrete(2),
         left:Discrete(2),
         nearbyCraft:Enum(furnace,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe),
         nearbySmelt:Enum(coal,iron_ingot,none),
         place:Enum(cobblestone,crafting_table,dirt,furnace,none,stone,torch),
         right:Discrete(2),
         sneak:Discrete(2),
         sprint:Discrete(2))

    It can be viewed as:
         - buttons, like attack, back, forward, sprint that are either pressed or not.
         - mouse, i.e. the continuous camera action in degrees. The two values are
           pitch (up/down), where up is negative, down is positive, and yaw (left/right),
           where left is negative, right is positive.
         - craft/equip/place actions for items specified above.
    So an example action could be sprint + forward + jump + attack + turn camera,
    all in one action.

    This wrapper makes the action space much smaller by selecting a few common actions
    and making the camera actions discrete.
    """

    def __init__(self, env, camera_angle=10, camera_noise=5):
        super().__init__(env)

        self.camera_angle = camera_angle
        self.camera_noise = camera_noise
        self._actions = [
            [('forward', 1)],  # 0
            [('back', 1)],  # 1
            [('left', 1)],  # 2
            [('right', 1)],  # 3
            [('jump', 1)],  # 4
            [('forward', 1), ('jump', 1)],  # 5
            [('camera', [-self.camera_angle, 0])],  # 6
            [('camera', [self.camera_angle, 0])],  # 7
            [('camera', [0, self.camera_angle])],  # 8
            [('camera', [0, -self.camera_angle])],  # 9
            [('attack', 1)],  # 10
            [('use', 1)],  # 11
            *[[('equip', item)]
              for item in ObservationSpace.items().keys()]
        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
            for a, v in actions:
                act[a] = v
            self.actions.append(act)

        self.action_space = gym.spaces.Discrete(len(self.actions))

        num_actions = self.action_space.n
        action_list = np.arange(num_actions)
        self.action_list = action_list

    def action(self, action):
        action = self.actions[action]
        action['camera'][0] += random.normal(scale=self.camera_noise)
        action['camera'][1] += random.normal(scale=self.camera_noise)
        return action
