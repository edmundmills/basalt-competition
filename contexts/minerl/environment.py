import gym
import minerl

import os
import copy
from collections import OrderedDict
import random

import numpy as np
import torch as th
import torch.nn.functional as F


environment_items = {'MineRLBasaltBuildVillageHouse-v0': OrderedDict([
    ("acacia_door", 64),
    ("acacia_fence", 64),
    ("cactus", 3),
    ("cobblestone", 64),
    ("dirt", 64),
    ("fence", 64),
    ("flower_pot", 3),
    ("glass", 64),
    ("ladder", 64),
    ("log#0", 64),
    ("log#1", 64),
    ("log2#0", 64),
    ("planks#0", 64),
    ("planks#1", 64),
    ("planks#4", 64),
    ("red_flower", 3),
    ("sand", 64),
    ("sandstone#0", 64),
    ("sandstone#2", 64),
    ("sandstone_stairs", 64),
    ("snowball", 1),
    ("spruce_door", 64),
    ("spruce_fence", 64),
    ("stone_axe", 1),
    ("stone_pickaxe", 1),
    ("stone_stairs", 64),
    ("torch", 64),
    ("wooden_door", 64),
    ("wooden_pressure_plate", 64)
]),
    'MineRLBasaltCreateVillageAnimalPen-v0': OrderedDict([
        ('fence', 64),
        ('fence_gate', 64),
        ('snowball', 1),
    ]),
    'MineRLBasaltFindCave-v0': OrderedDict([('snowball', 1)]),
    'MineRLBasaltMakeWaterfall-v0': OrderedDict([('water_bucket', 1),
                                                 ('snowball', 1)]),
    'MineRLTreechop-v0': OrderedDict([('snowball', 1)]),
    'MineRLNavigateExtremeDense-v0': OrderedDict([('compassAngle', 180)]),
    'MineRLNavigateDense-v0': OrderedDict([('compassAngle', 180)]),
}


class MinerlDebugEnv:
    def __init__(self, config):
        self.context = MineRLContext(config)
        self.action_list = self.context.actions
        self.inventory = self.context.starting_inventory

    def step(self, _action):
        obs = {"pov": np.random.randint(0, 255, (64, 64, 3)),
               "inventory": self.inventory,
               "compassAngle": 0,
               "equipped_items": {"mainhand": {'type': 'snowball'}}}
        _reward = 0
        _info = None
        done = np.random.choice([True, False], p=[.02, .98])
        return obs, _reward, done, _info

    def reset(self):
        obs, _, _, _ = self.step(0)
        return obs

    def close(self):
        return


class MineRLContext:
    def __init__(config):
        self.environment = config.env.name
        self.frame_shape = (3, 64, 64)
        self.items = list(environment_items[environment].keys())
        self.starting_inventory = environment_items[environment]
        self.action_name_list = ['Forward',  # 0
                                 'Back',  # 1
                                 'Left',  # 2
                                 'Right',  # 3
                                 'Jump',  # 4
                                 'Forward Jump',  # 5
                                 'Look Up',  # 6
                                 'Look Down',  # 7
                                 'Look Right',  # 8
                                 'Look Left',  # 9
                                 'Attack',  # 10
                                 'Use',  # 11
                                 'Equip']  # 12
        self.actions = list(range(len(self.action_name_list) - 1 + len(self.items)))
        if environment in ['MineRLTreechop-v0', 'MineRLNavigateDense-v0',
                           'MineRLNavigateExtremeDense-v0']:
            # no use or equip actions
            self.items_available = False
            self.action_name_list = self.action_name_list[:-2]
            self.actions = list(range(len(self.action_name_list)))
            self.use_action = -1
            self.equip_snowball_action = -1
            self.n_non_equip_actions = len(self.actions)
        else:
            self.items_available = True
            self.use_action = 11
            self.snowball_number = self.items.index('snowball')
            self.one_hot_snowball = F.one_hot(th.LongTensor([snowball_number]),
                                              len(self.items))
            self.equip_snowball_action = len(self.actions) - 1 + self.snowball_number
            n_non_equip_actions = len(self.action_name_list) - 1
            self.n_non_equip_actions = len(self.action_name_list) - 1

    def obs_to_pov(self, obs):
        pov = obs['pov'].copy()
        if isinstance(pov, np.ndarray):
            pov = th.from_numpy(pov).to(th.uint8)
        return pov.permute(2, 0, 1)

    def obs_to_equipped_item(self, obs):
        equipped_item = obs['equipped_items']['mainhand']['type']
        equipped = th.zeros(len(self.items), dtype=th.uint8)
        if equipped_item in items:
            equipped[items.index(equipped_item)] = 1
        return equipped

    def obs_to_inventory(self, obs):
        inventory = obs['inventory']
        first_item = list(inventory.values())[0]
        if isinstance(first_item, np.ndarray):
            inventory = {k: th.from_numpy(v).unsqueeze(0).to(th.uint8)
                         for k, v in inventory.items()}
        elif isinstance(first_item, (int, np.int32, np.int64)):
            inventory = {k: th.tensor([v], dtype=th.uint8) for k, v in inventory.items()}
        inventory = [inventory[item_name] for item_name in self.items]
        inventory = th.cat(inventory, dim=0)
        return inventory

    def obs_to_items(self, obs):
        if self.environment == 'MineRLTreechop-v0':
            items = th.zeros(2, dtype=th.uint8)
        elif self.environment in ['MineRLNavigateDense-v0',
                                  'MineRLNavigateExtremeDense-v0']:
            if 'compassAngle' in obs.keys():
                items = th.FloatTensor([obs['compassAngle'], 0])
            else:
                items = th.FloatTensor([float(obs['compass']['angle']), 0])
        else:
            items = th.cat((self.obs_to_inventory(obs),
                            self.obs_to_equipped_item(obs)), dim=0)
        return items

    def action_name(self, action_number):
        if action_number >= self.n_non_equip_actions:
            item = self.items[action_number - self.n_non_equip_actions]
            action_name = f'Equip {item}'
        else:
            action_name = self.action_name_list[action_number]
        return action_name

    def random_action(self):
        action = np.random.choice(self.actions)
        return action

    def equipped_item_name(self, state):
        if not self.items_available:
            return 'no_item'
        items = state[1]
        _inventory, equipped_item = th.chunk(items.reshape(1, -1), 2, dim=1)
        if th.sum(equipped_item).item() == 0:
            return 'no_item'
        equipped_item_number = equipped_item.squeeze(dim=0).nonzero()
        equipped_item_name = self.items[equipped_item_number.item()]
        return equipped_item_name

    def snowball_equipped(self, state, device='cpu'):
        if not self.items_available:
            return False
        items = state[1]
        _inventory, equipped_item = th.chunk(items.reshape(1, -1), 2, dim=1)
        snowball_equipped = th.all(th.eq(equipped_item,
                                         ActionSpace.one_hot_snowball().to(device)))
        return snowball_equipped.item()

    def threw_snowball(self, obs_or_state, action, device='cpu'):
        if not self.items_available:
            return False
        if isinstance(obs_or_state, dict):
            equipped_item = obs_or_state['equipped_items']['mainhand']['type']
        else:
            items = obs_or_state[1]
            _inventory, equipped_item = th.chunk(items.reshape(1, -1), 2, dim=1)
            if th.all(th.eq(equipped_item, ActionSpace.one_hot_snowball().to(device))):
                equipped_item = 'snowball'
        return action == 11 and equipped_item == 'snowball'

    def threw_snowball_list(self, obs, actions):
        if not self.items_available:
            return [False for action in actions]
        equipped_items = obs['equipped_items']['mainhand']['type']
        if isinstance(actions, th.Tensor):
            actions = actions.squeeze().tolist()
        return [item == 'snowball' and action == 11
                for item, action in zip(equipped_items, actions)]

    def threw_snowball_tensor(self, states, actions, device='cpu'):
        use_actions = th.eq(actions, 11).reshape(-1, 1)
        batch_size = use_actions.size()[0]
        snowball_tensor = self.one_hot_snowball.repeat(batch_size, 1).to(device)
        snowball_equipped = th.all(
            th.eq(th.chunk(states[1], 2, dim=1)[1], snowball_tensor), dim=1, keepdim=True)
        threw_snowball = use_actions * snowball_equipped
        return threw_snowball.type(th.uint8)


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

    def __init__(self, env, config):
        super().__init__(env)
        self.context = MineRLContext(config)
        self.camera_angle = config.env.camera_angle
        self.camera_noise = config.env.camera_noise
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
            *[[('equip', item)] for item in self.context.items]
        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.no_op()
            for a, v in actions:
                act[a] = v
            self.actions.append(act)

        self.action_space = gym.spaces.Discrete(len(self.actions))

        num_actions = self.action_space.n
        action_list = np.arange(num_actions)
        self.action_list = action_list

    def action(self, action):
        action = self.actions[action]
        action = copy.deepcopy(action)
        action['camera'][0] += np.random.normal(loc=0., scale=self.camera_noise)
        action['camera'][1] += np.random.normal(loc=0, scale=self.camera_noise)
        return action
