import copy
import random

import np
import torch as th
import torch.nn as nn
import kornia.augmentation as aug


class RandomHorizontalMirror:
    def __init__(self):
        pass

    def mirror_action(self, action):
        twos = action == 2
        threes = action == 3
        nines = action == 9
        tens = action == 10
        action = action + twos - threes + nines - tens
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
        noise = th.cat((th.randn((batch_size, int(item_dim / 2))) * self.inventory_noise,
                       th.zeros((batch_size, int(item_dim / 2)))), dim=1)
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


class RandomTranslate:
    def __init__(self):
        pass

    def random_translate(self, pov):
        new_pov = nn.ReplicationPad2d(4)(pov)
        new_pov = aug.RandomCrop((64, 64))(new_pov)
        return new_pov

    def __call__(self, sample):
        state, action, next_state, done, reward = sample
        pov, _items = state
        next_pov, _next_items = next_state
        new_state = self.random_translate(pov), _items
        new_next_state = self.random_translate(next_pov), _next_items
        sample = new_state, action, new_next_state, done, reward
        return sample


class PastFrameDropout:
    def __init__(self, dropout_frames):
        self.dropout_frames = dropout_frames

    def dropout_frames(self, pov):
        batch_size, frame_count, _, _ = pov.size()
        frame_count /= 3
        past_frame_count = frame_count - 1
        kept_frames = [idx + 1 for idx in sorted(random.sample(
            range(past_frame_count), past_frame_count - self.dropout_frames))]
        kept_frames = [0].append(kept_frames)
        with th.no_grad():
            frames = th.chunk(pov, frame_count, dim=1)
            new_pov = th.cat([frames[idx] for idx in kept_frames], dim=1)
        return new_pov

    def __call__(self, sample):
        state, action, next_state, done, reward = sample
        pov, _items = state
        next_pov, _next_items = next_state
        new_state = self.dropout_frames(pov), _items
        new_next_state = self.dropout_frames(next_pov), _next_items
        sample = new_state, action, new_next_state, done, reward
        return sample


class DataAgumentation:
    def __init__(self, config):
        self.transforms = []
        if config.dropout_frames > 0:
            self.transforms.append(DropoutFrames(config.dropout_frames))
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
