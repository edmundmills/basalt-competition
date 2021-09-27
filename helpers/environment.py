import os
import copy

import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import kornia.augmentation as aug


class EnvironmentHelper:
    environment_names = ['MineRLBasaltBuildVillageHouse-v0',
                         'MineRLBasaltCreateVillageAnimalPen-v0',
                         'MineRLBasaltFindCave-v0',
                         'MineRLBasaltMakeWaterfall-v0',
                         'MineRLTreechop-v0']


class ObservationSpace:
    environment_items = {'MineRLBasaltBuildVillageHouse-v0': {
        "acacia_door": 64,
        "acacia_fence": 64,
        "cactus": 3,
        "cobblestone": 64,
        "dirt": 64,
        "fence": 64,
        "flower_pot": 3,
        "glass": 64,
        "ladder": 64,
        "log#0": 64,
        "log#1": 64,
        "log2#0": 64,
        "planks#0": 64,
        "planks#1": 64,
        "planks#4": 64,
        "red_flower": 3,
        "sand": 64,
        "sandstone#0": 64,
        "sandstone#2": 64,
        "sandstone_stairs": 64,
        "snowball": 1,
        "spruce_door": 64,
        "spruce_fence": 64,
        "stone_axe": 1,
        "stone_pickaxe": 1,
        "stone_stairs": 64,
        "torch": 64,
        "wooden_door": 64,
        "wooden_pressure_plate": 64
    },
        'MineRLBasaltCreateVillageAnimalPen-v0': {
            'fence': 64,
            'fence_gate': 64,
            'snowball': 1,
    },
        'MineRLBasaltFindCave-v0': {'snowball': 1},
        'MineRLBasaltMakeWaterfall-v0': {'waterbucket': 1, 'snowball': 1},
        'MineRLTreechop-v0': {'snowball': 1},
    }

    frame_shape = (3, 64, 64)

    def items():
        environment = os.getenv('MINERL_ENVIRONMENT')
        return list(ObservationSpace.environment_items[environment].keys())

    def starting_inventory():
        environment = os.getenv('MINERL_ENVIRONMENT')
        return ObservationSpace.environment_items[environment]

    def obs_to_pov(obs):
        obs = obs['pov']
        if isinstance(obs, np.ndarray):
            obs = th.from_numpy(obs.copy())
        return obs.permute(2, 0, 1) / 255.0

    def obs_to_frame_sequence(obs):
        if 'frame_sequence' not in obs.keys():
            return None
        frame_sequence = obs['frame_sequence']
        if frame_sequence is None or frame_sequence.data[0] is None:
            return None
        return frame_sequence.permute(2, 0, 1) / 255.0

    def obs_to_equipped_item(obs):
        equipped_item = obs['equipped_items']['mainhand']['type']
        items = ObservationSpace.items()
        equipped = th.zeros((len(items))).long()
        if equipped_item in items:
            equipped[items.index(equipped_item)] = 1
        return equipped

    def obs_to_inventory(obs):
        inventory = obs['inventory']
        first_item = list(inventory.values())[0]
        if isinstance(first_item, np.ndarray):
            inventory = {k: th.from_numpy(v).unsqueeze(0) for k, v in inventory.items()}
        elif isinstance(first_item, (int, np.int32)):
            inventory = {k: th.LongTensor([v]) for k, v in inventory.items()}
        # normalize inventory by starting inventory
        inventory = [inventory[item_name] / starting_count
                     for item_name, starting_count
                     in iter(ObservationSpace.starting_inventory().items())]
        inventory = th.cat(inventory, dim=0)
        return inventory

    def obs_to_state(obs):
        pov = ObservationSpace.obs_to_pov(obs)
        frame_sequence = ObservationSpace.obs_to_frame_sequence(obs)
        if frame_sequence is not None:
            pov = th.cat((pov, frame_sequence), dim=0)
        environment = os.getenv('MINERL_ENVIRONMENT')
        if environment == 'MineRLTreechop-v0':
            items = th.zeros((2))
        else:
            items = th.cat((ObservationSpace.obs_to_inventory(obs),
                            ObservationSpace.obs_to_equipped_item(obs)), dim=0)
        state = (pov, items)
        return state


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

    def actions():
        actions = list(range(len(ActionSpace.action_name_list) - 1 +
                             len(ObservationSpace.items())))
        environment = os.getenv('MINERL_ENVIRONMENT')
        if environment == 'MineRLTreechop-v0':
            # no use or equip actions
            actions = actions[:-2]
        return actions

    def random_action():
        action = np.random.choice(ActionSpace.actions())
        return action

    def one_hot_snowball():
        snowball_number = ObservationSpace.items().index('snowball')
        return F.one_hot(th.LongTensor([snowball_number]), len(ObservationSpace.items()))

    def threw_snowball(obs_or_state, action):
        environment = os.getenv('MINERL_ENVIRONMENT')
        if environment == 'MineRLTreechop-v0':
            return False
        if isinstance(obs_or_state, dict):
            equipped_item = obs_or_state['equipped_items']['mainhand']['type']
        else:
            _pov, items = obs_or_state
            _inventory, equipped_item = th.chunk(items, 2, dim=1)
            if th.all(th.eq(equipped_item, ActionSpace.one_hot_snowball())):
                equipped_item = 'snowball'
        return action == 11 and equipped_item == 'snowball'

    def threw_snowball_list(obs, actions):
        environment = os.getenv('MINERL_ENVIRONMENT')
        if environment == 'MineRLTreechop-v0':
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
        if environment != 'MineRLTreechop-v0':
            equip_actions = dataset_actions["equip"]
            use_actions = dataset_actions["use"].reshape(-1)

        batch_size = len(attack_actions)
        actions = np.zeros((batch_size,), dtype=np.int)
        items = ObservationSpace.items()

        for i in range(batch_size):
            if environment != 'MineRLTreechop-v0' and use_actions[i] == 1:
                actions[i] = 11
            elif environment != 'MineRLTreechop-v0' and equip_actions[i] in items:
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


class RandomHorizontalMirror:
    def __init__(self):
        pass

    def mirror_action(self, action):
        if action == 2:
            new_action = 3
        elif action == 3:
            new_action = 2
        elif action == 9:
            new_action = 10
        elif action == 10:
            new_action = 9
        else:
            new_action = copy.copy(action)
        return new_action

    def mirror_pov(self, pov):
        new_pov = th.flip(pov, dims=[2])
        return new_pov

    def __call__(self, sample):
        if np.random.choice([True, False]):
            return sample

        state, action, next_state, done, reward = sample
        pov, _items = state
        next_pov, _next_items = next_state
        new_state = self.mirror_pov(pov), _items
        new_next_state = self.mirror_pov(next_pov), _next_items
        new_action = self.mirror_action(action)
        sample = new_state, new_action, new_next_state, done, reward
        return sample


class InventoryNoise:
    def __init__(self, inventory_noise):
        self.inventory_noise = inventory_noise

    def transform(self, items):
        item_dim = items.size()[0]
        with th.no_grad():
            noise = th.cat((th.randn((int(item_dim / 2))) * self.inventory_noise,
                           th.zeros((int(item_dim / 2)))), dim=0)
            new_items = th.clamp(items.clone() + noise, 0, 1)
        return new_items

    def __call__(self, sample):
        state, action, next_state, done, reward = sample
        _pov, items = state
        _next_pov, next_items = next_state
        new_state = _pov, self.transform(items)
        new_next_state = _next_pov, self.transform(next_items)
        sample = new_state, action, new_next_state, done, reward
        return sample


class RandomTranslate:
    def __init__(self):
        pass

    def random_translate(self, pov):
        transform = nn.Sequential(nn.ReplicationPad2d(4), aug.RandomCrop((64, 64)))
        with th.no_grad():
            new_pov = transform(pov.clone().unsqueeze(0)).squeeze()
        return new_pov

    def __call__(self, sample):
        state, action, next_state, done, reward = sample
        pov, _items = state
        next_pov, _next_items = next_state
        new_state = self.random_translate(pov), _items
        new_next_state = self.random_translate(next_pov), _next_items
        sample = new_state, action, new_next_state, done, reward
        return sample
