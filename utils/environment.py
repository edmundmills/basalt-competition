import gym
import minerl

import os
import copy
from collections import OrderedDict
import random

import numpy as np
import torch as th
import torch.nn.functional as F


def start_env(debug_env=False):
    if debug_env:
        env = DebugEnv()
    else:
        environment = os.getenv('MINERL_ENVIRONMENT')
        env = gym.make(environment)
        env = ActionShaping(env)
    return env


class DebugEnv:
    def __init__(self):
        self.action_list = ActionSpace.actions()
        self.inventory = ObservationSpace.starting_inventory()

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


class EnvironmentHelper:
    environment_names = ['MineRLBasaltBuildVillageHouse-v0',
                         'MineRLBasaltCreateVillageAnimalPen-v0',
                         'MineRLBasaltFindCave-v0',
                         'MineRLBasaltMakeWaterfall-v0',
                         'MineRLNavigateExtremeDense-v0',
                         'MineRLNavigateDense-v0',
                         'MineRLTreechop-v0']


class ObservationSpace:
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

    frame_shape = (3, 64, 64)

    def items():
        environment = os.getenv('MINERL_ENVIRONMENT')
        return list(ObservationSpace.environment_items[environment].keys())

    def starting_inventory():
        environment = os.getenv('MINERL_ENVIRONMENT')
        return ObservationSpace.environment_items[environment]

    def obs_to_pov(obs):
        pov = obs['pov'].copy()
        if isinstance(pov, np.ndarray):
            pov = th.from_numpy(pov).to(th.uint8)
        return pov.permute(2, 0, 1)

    def obs_to_equipped_item(obs):
        equipped_item = obs['equipped_items']['mainhand']['type']
        items = ObservationSpace.items()
        equipped = th.zeros(len(items), dtype=th.uint8)
        if equipped_item in items:
            equipped[items.index(equipped_item)] = 1
        return equipped

    def obs_to_inventory(obs):
        inventory = obs['inventory']
        first_item = list(inventory.values())[0]
        if isinstance(first_item, np.ndarray):
            inventory = {k: th.from_numpy(v).unsqueeze(0).to(th.uint8)
                         for k, v in inventory.items()}
        elif isinstance(first_item, (int, np.int32, np.int64)):
            inventory = {k: th.tensor([v], dtype=th.uint8) for k, v in inventory.items()}
        inventory = [inventory[item_name]
                     for item_name in iter(ObservationSpace.starting_inventory().keys())]
        inventory = th.cat(inventory, dim=0)
        return inventory

    def obs_to_items(obs):
        environment = os.getenv('MINERL_ENVIRONMENT')
        if environment == 'MineRLTreechop-v0':
            items = th.zeros(2, dtype=th.uint8)
        elif environment in ['MineRLNavigateDense-v0', 'MineRLNavigateExtremeDense-v0']:
            if 'compassAngle' in obs.keys():
                items = th.FloatTensor([obs['compassAngle'], 0])
            else:
                items = th.FloatTensor([float(obs['compass']['angle']), 0])
        else:
            items = th.cat((ObservationSpace.obs_to_inventory(obs),
                            ObservationSpace.obs_to_equipped_item(obs)), dim=0)
        return items


class ActionSpace:
    action_name_list = ['Forward',  # 0
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

    def action_name(action_number):
        n_non_equip_actions = len(ActionSpace.action_name_list) - 1
        if action_number >= n_non_equip_actions:
            item = ObservationSpace.items()[action_number - n_non_equip_actions]
            return f'Equip {item}'
        return ActionSpace.action_name_list[action_number]

    def equip_snowball_action():
        environment = os.getenv('MINERL_ENVIRONMENT')
        if environment in ['MineRLTreechop-v0', 'MineRLNavigateDense-v0',
                           'MineRLNavigateExtremeDense-v0']:
            action = -1
        if environment == 'MineRLBasaltBuildVillageHouse-v0':
            action = 32
        elif environment == 'MineRLBasaltCreateVillageAnimalPen-v0':
            action = 14
        elif environment == 'MineRLBasaltMakeWaterfall-v0':
            action = 13
        elif environment == 'MineRLBasaltFindCave-v0':
            action = 12
        print(ActionSpace.action_name(action))
        return action

    def use_action():
        return 11

    def actions():
        actions = list(range(len(ActionSpace.action_name_list) - 1 +
                             len(ObservationSpace.items())))
        environment = os.getenv('MINERL_ENVIRONMENT')
        if environment in ['MineRLTreechop-v0', 'MineRLNavigateDense-v0',
                           'MineRLNavigateExtremeDense-v0']:
            # no use or equip actions
            actions = actions[:-2]
        return actions

    def random_action():
        action = np.random.choice(ActionSpace.actions())
        return action

    def equipped_item(state):
        environment = os.getenv('MINERL_ENVIRONMENT')
        if environment in ['MineRLTreechop-v0', 'MineRLNavigateDense-v0',
                           'MineRLNavigateExtremeDense-v0']:
            return 'no_item'
        items = state[1]
        _inventory, equipped_item = th.chunk(items.reshape(1, -1), 2, dim=1)
        if th.eq(equipped_item, th.zeros(equipped_item.size())):
            return 'no_item'
        equipped_item_number = equipped_item.squeeze(dim=0).nonzero()
        equipped_item_name = ObservationSpace.items()[equipped_item_number.item()]
        return equipped_item_name

    def one_hot_snowball():
        snowball_number = ObservationSpace.items().index('snowball')
        return F.one_hot(th.LongTensor([snowball_number]), len(ObservationSpace.items()))

    def snowball_equipped(state, device='cpu'):
        environment = os.getenv('MINERL_ENVIRONMENT')
        if environment in ['MineRLTreechop-v0', 'MineRLNavigateDense-v0',
                           'MineRLNavigateExtremeDense-v0']:
            return False
        items = state[1]
        _inventory, equipped_item = th.chunk(items.reshape(1, -1), 2, dim=1)
        snowball_equipped = th.all(th.eq(equipped_item,
                                         ActionSpace.one_hot_snowball().to(device)))
        return snowball_equipped.item()

    def threw_snowball(obs_or_state, action, device='cpu'):
        environment = os.getenv('MINERL_ENVIRONMENT')
        if environment in ['MineRLTreechop-v0', 'MineRLNavigateDense-v0',
                           'MineRLNavigateExtremeDense-v0']:
            return False
        if isinstance(obs_or_state, dict):
            equipped_item = obs_or_state['equipped_items']['mainhand']['type']
        else:
            items = obs_or_state[1]
            _inventory, equipped_item = th.chunk(items.reshape(1, -1), 2, dim=1)
            if th.all(th.eq(equipped_item, ActionSpace.one_hot_snowball().to(device))):
                equipped_item = 'snowball'
        return action == 11 and equipped_item == 'snowball'

    def threw_snowball_list(obs, actions):
        environment = os.getenv('MINERL_ENVIRONMENT')
        if environment in ['MineRLTreechop-v0', 'MineRLNavigateDense-v0',
                           'MineRLNavigateExtremeDense-v0']:
            return [False for action in actions]
        equipped_items = obs['equipped_items']['mainhand']['type']
        if isinstance(actions, th.Tensor):
            actions = actions.squeeze().tolist()
        return [item == 'snowball' and action == 11
                for item, action in zip(equipped_items, actions)]

    def threw_snowball_tensor(states, actions, device):
        use_actions = th.eq(actions, 11).reshape(-1, 1)
        batch_size = use_actions.size()[0]
        snowball_tensor = ActionSpace.one_hot_snowball().repeat(batch_size, 1).to(device)
        snowball_equipped = th.all(
            th.eq(th.chunk(states[1], 2, dim=1)[1], snowball_tensor), dim=1, keepdim=True)
        threw_snowball = use_actions * snowball_equipped
        return threw_snowball.type(th.uint8)

    def dataset_action_batch_to_actions(dataset_actions, camera_margin=5):
        """
        Turn a batch of actions from dataset to a numpy
        array that corresponds to batch of actions of ActionShaping wrapper (_actions).

        Camera margin sets the threshold what is considered "moving camera".

        Array elements are integers corresponding to actions, or "-1"
        for actions that did not have any corresponding discrete match.
        """
        camera_actions = dataset_actions["camera"].reshape((-1, 2))
        attack_actions = dataset_actions["attack"].reshape(-1)
        forward_actions = dataset_actions["forward"].reshape(-1)
        back_actions = dataset_actions["back"].reshape(-1)
        left_actions = dataset_actions["left"].reshape(-1)
        right_actions = dataset_actions["right"].reshape(-1)
        jump_actions = dataset_actions["jump"].reshape(-1)
        environment = os.getenv('MINERL_ENVIRONMENT')
        if environment not in ['MineRLTreechop-v0', 'MineRLNavigateDense-v0',
                               'MineRLNavigateExtremeDense-v0']:
            equip_actions = dataset_actions["equip"]
            use_actions = dataset_actions["use"].reshape(-1)

        batch_size = len(attack_actions)
        actions = np.zeros((batch_size,), dtype=np.int)
        items = ObservationSpace.items()

        for i in range(batch_size):
            if environment not in ['MineRLTreechop-v0', 'MineRLNavigateDense-v0',
                                   'MineRLNavigateExtremeDense-v0'] \
                    and use_actions[i] == 1:
                actions[i] = 11
            elif environment not in ['MineRLTreechop-v0', 'MineRLNavigateDense-v0',
                                     'MineRLNavigateExtremeDense-v0'] \
                    and equip_actions[i] in items:
                actions[i] = 12 + items.index(equip_actions[i])
            elif camera_actions[i][0] < -camera_margin:
                actions[i] = 6
            elif camera_actions[i][0] > camera_margin:
                actions[i] = 7
            elif camera_actions[i][1] > camera_margin:
                actions[i] = 8
            elif camera_actions[i][1] < -camera_margin:
                actions[i] = 9
            elif forward_actions[i] == 1 and jump_actions[i] == 1:
                actions[i] = 5
            elif forward_actions[i] == 1 and jump_actions[i] != 1:
                actions[i] = 0
            elif attack_actions[i] == 1:
                actions[i] = 10
            elif jump_actions[i] == 1:
                actions[i] = 4
            elif back_actions[i] == 1:
                actions[i] = 1
            elif left_actions[i] == 1:
                actions[i] = 2
            elif right_actions[i] == 1:
                actions[i] = 3
            else:
                actions[i] = -1
        return actions


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

    def __init__(self, env, camera_angle=10, camera_noise=0.5):
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
              for item in ObservationSpace.items()]
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
