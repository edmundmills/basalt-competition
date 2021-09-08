from helpers.environment import ObservationSpace, MirrorAugmentation

from pathlib import Path
import os
from collections import deque
import json
import copy

import torch as th
import math
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader


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
        trajectory_lengths = []
        for environment_name in self.environments:
            environment_path = self.data_root / environment_name
            paths, lengths = self._get_step_data(environment_path)
            step_paths.extend(paths)
            trajectory_lengths.extend(lengths)
        self.step_paths = step_paths
        self.trajectory_lengths = trajectory_lengths

    def _get_step_data(self, environment_path):
        step_paths = []
        trajectory_lengths = []
        for trajectory_path in sorted(environment_path.iterdir()):
            steps_dir_path = trajectory_path / 'steps'
            if not steps_dir_path.is_dir():
                continue
            trajectory_step_paths = sorted(steps_dir_path.iterdir())
            trajectory_length = len(trajectory_step_paths)
            step_paths.extend(trajectory_step_paths)
            trajectory_lengths.append(trajectory_length)
        return step_paths, trajectory_lengths

    def trajectory_length(self, step_path):
        trajectory_path = step_path.parent.parent
        trajectory_index = self.trajectory_paths.index(trajectory_path)
        length = self.trajectory_lengths[trajectory_index]
        return length

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

    def _load_step_dict(self, idx):
        step_path = self.step_paths[idx]
        step_dict = np.load(step_path, allow_pickle=True).item()
        return step_dict


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

    def _frame_sequence(self, step_number, idx_in_dataset):
        initial_step_idx = idx_in_dataset - step_number
        frame_indices = [max(initial_step_idx, idx_in_dataset - 1 - frame_number)
                         for frame_number in reversed(range(self.number_of_frames - 1))]
        frame_sequence = np.array([self._load_step_dict(frame_idx)['obs']['pov']
                                   for frame_idx in frame_indices])
        return frame_sequence


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque([], maxlen=int(capacity))

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

    def push(self, obs, action, next_obs, done):
        self.buffer.append((copy.deepcopy(obs), action.copy(),
                            copy.deepcopy(next_obs), done))

    def sample(self, batch_size):
        replay_batch_size = min(batch_size, len(self.buffer))
        dataloader = iter(DataLoader(self,
                                     shuffle=True,
                                     batch_size=replay_batch_size))
        (replay_obs, replay_actions, replay_next_obs,
            replay_done) = next(dataloader)
        return replay_obs, replay_actions, replay_next_obs, replay_done


class MixedReplayBuffer(ReplayBuffer):
    '''
    Samples a fraction from the expert trajectories
    and the remainder from the replay buffer.
    '''

    def __init__(self,
                 capacity=1e6,
                 batch_size=64,
                 expert_sample_fraction=0.5):
        self.batch_size = batch_size
        self.expert_sample_fraction = expert_sample_fraction
        self.expert_batch_size = math.floor(batch_size * self.expert_sample_fraction)
        self.replay_batch_size = self.batch_size - self.expert_batch_size
        super().__init__(capacity)
        self.expert_dataset = MultiFrameDataset()
        self.expert_dataloader = self._initialize_dataloader()

    def _initialize_dataloader(self):
        return iter(DataLoader(self.expert_dataset,
                               shuffle=True,
                               batch_size=self.expert_batch_size,
                               drop_last=True))

    def sample_replay(self):
        return self.sample(self.replay_batch_size)

    def sample_expert(self):
        try:
            (expert_obs, expert_actions, expert_next_obs,
                expert_done) = next(self.expert_dataloader)
        except StopIteration:
            self.expert_dataloader = self._initialize_dataloader()
            (expert_obs, expert_actions, expert_next_obs,
                expert_done) = next(self.expert_dataloader)
        return expert_obs, expert_actions, expert_next_obs, expert_done
