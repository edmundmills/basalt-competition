from core.state import State, Transition

from collections import OrderedDict, deque
import copy

import gym
import minerl
from omegaconf import OmegaConf
import numpy as np
import torch as th
import torch.nn.functional as F


pov_normalization_factors = {
    'MineRLBasaltBuildVillageHouse-v0': (
        th.FloatTensor([109.01, 108.78, 95.13]),
        th.FloatTensor([50.83, 56.32, 77.89])),
    'MineRLBasaltCreateVillageAnimalPen-v0': (
        th.FloatTensor([107.61, 125.33, 112.16]),
        th.FloatTensor([43.69, 50.70, 93.10])),
    'MineRLBasaltFindCave-v0': (
        th.FloatTensor([106.44, 127.52, 126.61]),
        th.FloatTensor([45.06, 54.25, 97.68])),
    'MineRLBasaltMakeWaterfall-v0': (
        th.FloatTensor([109.11, 117.04, 131.58]),
        th.FloatTensor([51.78, 60.46, 85.87])),
    'MineRLNavigateExtremeDense-v0': (
        th.FloatTensor([70.69, 71.73, 88.11]),
        th.FloatTensor([43.07, 49.05, 72.84])),
    'MineRLTreechop-v0': (
        th.FloatTensor([35.60, 51.53, 24.05]),
        th.FloatTensor([29.58, 38.48, 25.91])),
    'MineRLNavigateDense-v0': (
        th.FloatTensor([66.89, 76.77, 104.20]),
        th.FloatTensor([54.26, 60.83, 88.11])),
}

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


def start_env(config: OmegaConf, debug_env=False) -> gym.Env:
    """Starts the chosen environment and applies action and observation wrappers."""
    if debug_env:
        env = MineRLDebugEnv(config)
    else:
        environment = config.env.name
        env = gym.make(environment)
        env = ActionShaping(env, config)
    env = ObservationWrapper(env, config)
    return env


class MineRLDebugEnv(gym.Env):
    """Simulates a MineRL environment to reduce debug time."""

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

class Context(object):
    """Contexts inherit from this class."""

class MineRLContext(Context):
    """
    Processes the raw config and provides helper functions for the context.
    
    Stores information about the inventory, normaliztion factors, voluntary termination,
    state space dimensionality.
    """

    def __init__(self, config):
        self.environment = config.env.name
        self.frame_shape = (3, 64, 64)
        self.spatial_normalization = pov_normalization_factors[self.environment]
        self.items = list(environment_items[self.environment].keys())
        self.starting_inventory = environment_items[self.environment]
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
        if self.environment in ['MineRLTreechop-v0', 'MineRLNavigateDense-v0',
                                'MineRLNavigateExtremeDense-v0']:
            # no use or equip actions
            self.items_available = False
            self.action_name_list = self.action_name_list[:-2]
            self.actions = list(range(len(self.action_name_list)))
            self.use_action = -1
            self.n_non_equip_actions = len(self.actions)
            self.voluntary_termination = False
        else:
            self.items_available = True
            self.use_action = 11
            self.n_non_equip_actions = len(self.action_name_list) - 1
            self.voluntary_termination = True
        starting_count = th.FloatTensor(
            list(self.starting_inventory.values())).reshape(1, -1)
        ones = th.ones(starting_count.size())
        self.nonspatial_normalization = th.cat((starting_count, ones), dim=1)
        self.nonspatial_size = self.nonspatial_normalization.size()[1]
        self.lstm_hidden_size = config.model.lstm_hidden_size
        self.initial_hidden = th.zeros(self.lstm_hidden_size*2)
        self.termination_helper = TerminationHelper(self, config)

    def action_name(self, action_number):
        if action_number >= self.n_non_equip_actions:
            item = self.items[action_number - self.n_non_equip_actions]
            action_name = f'Equip {item}'
        else:
            action_name = self.action_name_list[action_number]
        return action_name

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


class TerminationHelper:
    def __init__(self, context, config):
        self.context = context
        if self.context.items_available:
            self.snowball_number = self.context.items.index('snowball')
            self.one_hot_snowball = F.one_hot(th.LongTensor([self.snowball_number]),
                                              len(self.context.items))
            self.equip_snowball_action = len(self.context.actions) - 1 \
                + self.snowball_number
        else:
            self.equip_snowball_action = -1
        self.suppress_voluntary_termination_steps = \
            config.context.suppress_voluntary_termination_steps

    def snowball_equipped(self, state):
        if not self.context.items_available:
            return False
        nonspatial = state.nonspatial
        _inventory, equipped_item = th.chunk(nonspatial.reshape(1, -1), 2, dim=1)
        snowball_equipped = th.all(th.eq(
            equipped_item, self.one_hot_snowball.to(state.nonspatial.device)))
        return snowball_equipped.item()

    def terminated(self, state, action):
        return self.snowball_equipped(state) and action == self.context.use_action

    def threw_snowball_tensor(self, states, actions, device='cpu'):
        use_actions = th.eq(actions, self.context.use_action).reshape(-1, 1)
        batch_size = use_actions.size()[0]
        snowball_tensor = self.one_hot_snowball.repeat(batch_size, 1).to(device)
        _items, equipped = th.chunk(states[1], 2, dim=1)
        snowball_equipped = th.all(th.eq(equipped, snowball_tensor), dim=1, keepdim=True)
        threw_snowball = use_actions * snowball_equipped
        return threw_snowball.type(th.uint8)

    def suppressed_termination(self, step, state, action):
        if step == 0 and self.suppress_voluntary_termination_steps > 0:
            print(('Suppressing voluntary termination for'
                   f' {self.suppress_voluntary_termination_steps} steps'))
        elif step == self.suppress_voluntary_termination_steps and step != 0:
            print('No longer suppressing voluntary termination')
        suppressed_termination = step < self.suppress_voluntary_termination_steps \
            and self.terminated(state, action)
        return suppressed_termination


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.context = MineRLContext(config)
        self.n_observation_frames = config.model.n_observation_frames
        self.framestack = deque(maxlen=self.n_observation_frames)

    def _obs_to_spatial(self, obs):
        pov = obs['pov'].copy()
        if isinstance(pov, np.ndarray):
            pov = th.from_numpy(pov).to(th.uint8)
        return pov.permute(2, 0, 1)

    def _obs_to_equipped_item(self, obs):
        equipped_item = obs['equipped_items']['mainhand']['type']
        equipped = th.zeros(len(self.context.items), dtype=th.uint8)
        if equipped_item in self.context.items:
            equipped[self.context.items.index(equipped_item)] = 1
        return equipped

    def _obs_to_inventory(self, obs):
        inventory = obs['inventory']
        first_item = list(inventory.values())[0]
        if isinstance(first_item, np.ndarray):
            inventory = {k: th.from_numpy(v).unsqueeze(0).to(th.uint8)
                         for k, v in inventory.items()}
        elif isinstance(first_item, (int, np.int32, np.int64)):
            inventory = {k: th.tensor([v], dtype=th.uint8) for k, v in inventory.items()}
        inventory = [inventory[item_name] for item_name in self.context.items]
        inventory = th.cat(inventory, dim=0)
        return inventory

    def _obs_to_nonspatial(self, obs):
        if self.context.environment == 'MineRLTreechop-v0':
            nonspatial = th.zeros(2, dtype=th.uint8)
        elif self.context.environment in ['MineRLNavigateDense-v0',
                                          'MineRLNavigateExtremeDense-v0']:
            if 'compassAngle' in obs.keys():
                nonspatial = th.FloatTensor([obs['compassAngle'], 0])
            else:
                nonspatial = th.FloatTensor([float(obs['compass']['angle']), 0])
        else:
            nonspatial = th.cat((self._obs_to_inventory(obs),
                                 self._obs_to_equipped_item(obs)), dim=0)
        return nonspatial

    def observation(self, obs):
        state = State(self._obs_to_spatial(obs),
                      self._obs_to_nonspatial(obs),
                      self.context.initial_hidden)
        pov = state.spatial
        while len(self.framestack) < self.n_observation_frames:
            self.framestack.append(pov)
        self.framestack.append(pov)
        spatial = th.cat(list(self.framestack), dim=0)
        state = list(state)
        state[0] = spatial
        state = State(*state)
        return state


class ActionShaping(gym.ActionWrapper):
    """
    This wrapper is based on the one provided in the competition baseline here:
    https://colab.research.google.com/drive/1qfjHCQkukFcR9w1aPvGJyQxOa-Gv7Gt_?usp=sharing
    
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
        self.camera_angle = config.context.camera_angle
        self.camera_noise = config.context.camera_noise
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
