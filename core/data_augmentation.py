from core.state import State, Transition

import copy
import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, pixels=4):
        self.transform = RandomShiftsAug(pixels)

    def random_translate(self, spatial):
        *n, c, h, w = spatial.size()
        spatial = spatial.reshape(-1, c, h, w)
        new_spatial = self.transform(spatial).reshape(*n, c, h, w)
        return new_spatial

    def __call__(self, transition):
        state, action, reward, next_state, done = transition
        state = list(state)
        next_state = list(next_state)
        state[0] = self.random_translate(state[0])
        if len(next_state) != 0:
            next_state[0] = self.random_translate(next_state[0])
        transition = Transition(State(*state), action, reward, State(*next_state), done)
        return transition


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

    def __call__(self, transition):
        if np.random.choice([True, False]):
            return transition

        state, action, reward, next_state, done = transition
        state = list(state)
        next_state = list(next_state)
        state[0] = self.mirror_pov(state[0])
        if len(next_state) != 0:
            next_state[0] = self.mirror_pov(next_state[0])
        new_action = self.mirror_action(action)
        transition = Transition(State(*state), new_action,
                                reward, State(*next_state), done)
        return transition


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

    def __call__(self, transition):
        state, action, reward, next_state, done = transition
        state = list(state)
        next_state = list(next_state)
        state[1] = self.transform(state[1])
        if len(next_state) != 0:
            next_state[1] = self.transform(next_state[1])
        transition = Transition(State(*state), action, reward, State(*next_state), done)
        return transition


class DataAugmentation:
    def __init__(self, config):
        self.transforms = []
        if config.context.mirror_augment:
            self.transforms.append(RandomHorizontalMirror())
        if config.context.random_translate_pixels > 0:
            self.transforms.append(RandomTranslate(
                config.context.random_translate_pixels))
        if config.context.inventory_noise > 0:
            self.transforms.append(InventoryNoise(config.context.inventory_noise))

    def __call__(self, transition):
        for transform in self.transforms:
            transition = transform(transition)
        return transition
