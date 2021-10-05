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
        new_pov = th.flip(pov, dims=[3])
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
        batch_size, item_dim = items.size()
        noise = th.randn((batch_size, int(item_dim / 2)), device=items.device) \
            * self.inventory_noise
        zeros = th.zeros((batch_size, int(item_dim / 2)), device=items.device)
        noise = th.cat((noise, zeros), dim=1)
        new_items = th.clamp(items + noise, 0, 1)
        return new_items

    def __call__(self, sample):
        state, action, next_state, done, reward = sample
        _pov, items = state
        _next_pov, next_items = next_state
        new_state = _pov, self.transform(items)
        new_next_state = _next_pov, self.transform(next_items)
        sample = new_state, action, new_next_state, done, reward
        return sample


class RandomShiftsAug(nn.Module):
    '''
    https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
    '''

    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = th.linspace(-1.0 + eps,
                             1.0 - eps,
                             h + 2 * self.pad,
                             device=x.device,
                             dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = th.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = th.randint(0,
                           2 * self.pad + 1,
                           size=(n, 1, 1, 2),
                           device=x.device,
                           dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class RandomTranslate:
    def __init__(self):
        self.transform = RandomShiftsAug(4)

    def random_translate(self, pov):
        new_pov = self.transform(pov)
        return new_pov

    def __call__(self, sample):
        state, action, next_state, done, reward = sample
        pov, _items = state
        next_pov, _next_items = next_state
        new_state = self.random_translate(pov), _items
        new_next_state = self.random_translate(next_pov), _next_items
        sample = new_state, action, new_next_state, done, reward
        return sample


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

    def __call__(self, state):
        for transform in self.transforms:
            state = transform(state)
        return state
