import os

import numpy as np
import torch as th
import torch.nn.functional as F


environment_names = ['MineRLBasaltBuildVillageHouse-v0',
                     'MineRLBasaltCreateVillageAnimalPen-v0',
                     'MineRLBasaltFindCave-v0',
                     'MineRLBasaltMakeWaterfall-v0']


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
    }

    frame_shape = (3, 64, 64)

    number_of_frames = 4

    def items():
        environment = os.getenv('MINERL_ENVIRONMENT')
        return ObservationSpace.environment_items[environment]

    def pov_tensor_from_single_obs(obs):
        return th.from_numpy(obs['pov'].copy()).unsqueeze(0).permute(0, 3, 1, 2).float()

    def dataset_obs_batch_to_pov(obs):
        return obs['pov'].permute(0, 3, 1, 2).float()

    def dataset_obs_batch_to_frame_sequence(obs):
        return obs['frame_sequence'].permute(0, 1, 4, 2, 3).float()

    def dataset_obs_batch_to_equipped(obs):
        equipped_item = obs['equipped_items']['mainhand']['type']
        items = list(ObservationSpace.items().keys())
        equipped = th.zeros((len(equipped_item), len(items))).long()
        for idx, item in enumerate(equipped_item):
            if item not in items:
                continue
            equipped[idx, items.index(item)] = 1
        return equipped

    def dataset_obs_batch_to_inventory(obs):
        inventory = obs['inventory']
        # normalize inventory by starting inventory
        inventory = th.cat([inventory[item].unsqueeze(1) / ObservationSpace.items()[item]
                            for item in ObservationSpace.items().keys()], dim=1)
        return inventory


class ActionSpace:
    def mirror_action(action):
        if action == 3:
            action = 4
        elif action == 4:
            action = 3
        elif action == 12:
            action = 11
        elif action == 11:
            action = 12
        return action

    action_name_list = ['Forward', 'Back', 'Left', 'Right',
                        'Jump', 'Forward Jump',
                        'Look Up', 'Look Down', 'Look Right', 'Look Left',
                        'Attack', 'Use', 'Equip']

    def actions():
        actions = list(range(len(ActionSpace.action_name_list) - 1 +
                             len(ObservationSpace.items().keys())))
        return actions

    def dataset_action_batch_to_actions(dataset_actions, camera_margin=5):
        """
        Turn a batch of actions from dataset to a numpy
        array that corresponds to batch of actions of ActionShaping wrapper (_actions).

        Camera margin sets the threshold what is considered "moving camera".

        Array elements are integers corresponding to actions, or "-1"
        for actions that did not have any corresponding discrete match.
        """
        camera_actions = dataset_actions["camera"].squeeze()
        attack_actions = dataset_actions["attack"].squeeze()
        forward_actions = dataset_actions["forward"].squeeze()
        back_actions = dataset_actions["back"].squeeze()
        left_actions = dataset_actions["left"].squeeze()
        right_actions = dataset_actions["right"].squeeze()
        jump_actions = dataset_actions["jump"].squeeze()
        equip_actions = dataset_actions["equip"]
        use_actions = dataset_actions["use"].squeeze()

        batch_size = len(camera_actions)
        actions = np.zeros((batch_size,), dtype=np.int)
        items = list(ObservationSpace.items().keys())

        for i in range(batch_size):
            if use_actions[i] == 1:
                actions[i] = 11
            elif equip_actions[i] in items:
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
