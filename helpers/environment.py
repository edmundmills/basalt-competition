import gym
import numpy as np


import torch as th
import torchvision
from torchvision.models.mobilenetv3 import mobilenet_v3_small
from torch import nn
import torch.nn.functional as F
import numpy as np

environment_names = ['MineRLBasaltBuildVillageHouse-v0',
                     'MineRLBasaltCreateVillageAnimalPen-v0',
                     'MineRLBasaltFindCave-v0',
                     'MineRLBasaltMakeWaterfall-v0']


def pov_tensor_from_single_obs(obs):
    return th.from_numpy(obs['pov'].copy()).unsqueeze(0).permute(0, 3, 1, 2).float()


def dataset_obs_batch_to_obs(obs):
    return obs['pov'].permute(0, 3, 1, 2).float()


action_name_list = ['Attack', 'Forward', 'Back', 'Left', 'Right', 'Jump', 'Use', 'Equip',
                    'Forward Jump', 'Look Up', 'Look Down', 'Look Right', 'Look Left']


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

    for i in range(len(camera_actions)):
        if use_actions[i] == 1:
            actions[i] = 6
        elif equip_actions[i] == 1:
            actions[i] = 7
        elif camera_actions[i][0] < -camera_margin:
            actions[i] = 9
        elif camera_actions[i][0] > camera_margin:
            actions[i] = 10
        elif camera_actions[i][1] > camera_margin:
            actions[i] = 11
        elif camera_actions[i][1] < -camera_margin:
            actions[i] = 12
        elif forward_actions[i] == 1 and jump_actions[i] == 1:
            actions[i] = 8
        elif forward_actions[i] == 1 and jump_actions[i] != 1:
            actions[i] = 1
        elif attack_actions[i] == 1:
            actions[i] = 0
        elif jump_actions[i] == 1:
            actions[i] = 5
        elif back_actions[i] == 1:
            actions[i] = 2
        elif left_actions[i] == 1:
            actions[i] = 3
        elif right_actions[i] == 1:
            actions[i] = 4
        else:
            actions[i] = -1
    return actions
