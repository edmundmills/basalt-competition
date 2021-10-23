from core.data_augmentation import RandomTranslate

import copy
import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class RandomHorizontalMirror:
    def __init__(self):
        pass

    def mirror_action(self, action):
        twos = action == 2
        threes = action == 3
        nines = action == 9
        tens = action == 10
        action = action + twos.int() - threes.int() + nines.int() - tens.int()
        return action

    def mirror_pov(self, pov):
        new_pov = th.flip(pov, dims=[-1])
        return new_pov

    def __call__(self, sample):
        if np.random.choice([True, False]):
            return sample

        state, action, next_state, done, reward = sample
        state = list(state)
        next_state = list(next_state)
        state[0] = self.mirror_pov(state[0])
        if len(next_state) != 0:
            next_state[0] = self.mirror_pov(next_state[0])
        new_action = self.mirror_action(action)
        sample = tuple(state), action, tuple(next_state), done, reward
        return sample


class InventoryNoise:
    def __init__(self, inventory_noise):
        self.inventory_noise = inventory_noise

    def transform(self, items):
        *batch_size, item_dim = items.size()
        noise = th.randn((*batch_size, int(item_dim / 2)), device=items.device) \
            * self.inventory_noise
        zeros = th.zeros((*batch_size, int(item_dim / 2)), device=items.device)
        noise = th.cat((noise, zeros), dim=-1)
        new_items = th.clamp(items + noise, 0, 1)
        return new_items

    def __call__(self, sample):
        state, action, next_state, done, reward = sample
        state = list(state)
        next_state = list(next_state)
        state[1] = self.transform(state[1])
        if len(next_state) != 0:
            next_state[1] = self.transform(next_state[1])
        sample = tuple(state), action, tuple(next_state), done, reward
        return sample


class DataAugmentation:
    def __init__(self, config):
        self.transforms = []
        if config.mirror_augment:
            self.transforms.append(RandomHorizontalMirror())
        if config.random_translate:
            self.transforms.append(RandomTranslate())
        if config.inventory_noise > 0:
            self.transforms.append(InventoryNoise(config.inventory_noise))

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample
