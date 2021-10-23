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
    def __init__(self):
        self.transform = RandomShiftsAug(4)

    def random_translate(self, pov):
        *n, c, h, w = pov.size()
        pov = pov.reshape(-1, c, h, w)
        new_pov = self.transform(pov).reshape(*n, c, h, w)
        return new_pov

    def __call__(self, sample):
        state, action, next_state, done, reward = sample
        state = list(state)
        next_state = list(next_state)
        state[0] = self.random_translate(state[0])
        if len(next_state) != 0:
            next_state[0] = self.random_translate(next_state[0])
        sample = tuple(state), action, tuple(next_state), done, reward
        return sample
