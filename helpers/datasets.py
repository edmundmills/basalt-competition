from helpers.environment import ObservationSpace, MirrorAugmentation

from pathlib import Path
import os
from collections import deque

import torch as th
import math
import random
import numpy as np

from torch.utils.data import Dataset


class StepDataset(Dataset):
    def __init__(self,
                 data_root=os.getenv('MINERL_DATA_ROOT'),
                 environments=[],
                 transform=MirrorAugmentation()):
        self.environments = environments if environments != [] else [
            os.getenv('MINERL_ENVIRONMENT')]
        self.data_root = Path(data_root)
        self.transform = transform
        step_paths = []
        for environment_name in self.environments:
            environment_path = self.data_root / environment_name
            step_paths.extend(self._get_step_data(environment_path))
        self.step_paths = step_paths

    def _get_step_data(self, environment_path):
        step_paths = []
        for trajectory_path in sorted(environment_path.iterdir()):
            steps_dir_path = trajectory_path / 'steps'
            if not steps_dir_path.is_dir():
                continue
            trajecory_step_paths = sorted(steps_dir_path.iterdir())
            step_paths.extend(trajecory_step_paths)
        return step_paths

    def __len__(self):
        return len(self.step_paths)

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()
        if idx + 1 > len(self):
            raise IndexError('list index out of range')
        step_dict = self._load_step_dict(idx)
        if step_dict['done']:
            next_obs = step_dict['obs']
        else:
            next_step_dict = self._load_step_dict(idx + 1)
            next_obs = next_step_dict['obs']
        sample = (step_dict['obs'], step_dict['action'], next_obs, step_dict['done'])
        if self.transform:
            sample = self.transform(sample)
        return sample


class MultiFrameDataset(StepDataset):
    def __init__(self,
                 data_root=os.getenv('MINERL_DATA_ROOT'),
                 environments=[],
                 transform=MirrorAugmentation()):
        super().__init__(data_root, environments, transform)
        self.number_of_frames = ObservationSpace.number_of_frames

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()
        if idx + 1 > len(self):
            raise IndexError('list index out of range')
        step_dict = self._load_step_dict(idx)
        step_dict['obs']['frame_sequence'] = self._frame_sequence(step_dict['step'], idx)
        if step_dict['done']:
            next_obs = step_dict['obs']
        else:
            next_step_dict = self._load_step_dict(idx + 1)
            next_step_dict['obs']['frame_sequence'] = self._frame_sequence(
                next_step_dict['step'], idx + 1)
            next_obs = next_step_dict['obs']
        sample = (step_dict['obs'], step_dict['action'], next_obs, step_dict['done'])
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _load_step_dict(self, idx):
        step_path = self.step_paths[idx]
        step_dict = np.load(step_path, allow_pickle=True).item()
        return step_dict

    def _frame_sequence(self, step_number, idx_in_dataset):
        frame_indices = [idx_in_dataset - step_number +
                         int(math.floor(step_number *
                                        frame_number / (self.number_of_frames - 1)))
                         for frame_number in range(self.number_of_frames - 1)]
        frame_sequence = np.array([self._load_step_dict(frame_idx)['obs']['pov']
                                   for frame_idx in frame_indices])
        return frame_sequence


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, next_state, done):
        self.buffer.append((state, action, next_state, done))

    def sample(self, batch_size):
        replay_batch_size = min(batch_size, len(self.buffer))
        replay_batch = random.sample(self.buffer, replay_batch_size)
        (replay_states, replay_actions,
         replay_next_states, replay_done) = zip(*replay_batch)
        replay_states = th.cat(replay_states, dim=0)
        replay_actions = th.LongTensor(replay_actions).unsqueeze(1)
        replay_next_states = th.cat(replay_states, dim=0)
        replay_done = th.LongTensor(replay_done).unsqueeze(1)
        return replay_states, replay_actions, replay_next_states, replay_done
