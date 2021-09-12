from helpers.environment import ObservationSpace, MirrorAugmentation
from helpers.trajectories import Trajectory

import minerl

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
from torch.utils.data.dataloader import default_collate


class TrajectoryStepDataset(Dataset):
    def __init__(self,
                 transform=None,
                 multiframe=False):
        self.data_root = Path(os.getenv('MINERL_DATA_ROOT'))
        self.environment = os.getenv('MINERL_ENVIRONMENT')
        self.multiframe = multiframe
        self.environment_path = self.data_root / self.environment
        self.transform = transform
        self.trajectories, self.step_lookup = self._load_data()

    def _load_data(self):
        data = minerl.data.make(self.environment)
        trajectories = []
        step_lookup = []
        trajectory_paths = self.environment_path.iterdir()
        # trajectory_paths = [self.environment_path /
        #                     'v3_few_grapefruit_medusa-4_8382-15666']
        for trajectory_idx, trajectory_path in enumerate(trajectory_paths):
            if not trajectory_path.is_dir():
                continue

            trajectory = Trajectory(path=trajectory_path)
            for step_idx, (obs, action, _, _, done) \
                    in enumerate(data.load_data(str(trajectory_path))):
                trajectory.obs.append(obs)
                trajectory.actions.append(action)
                trajectory.done = done
                step_lookup.append((trajectory_idx, step_idx))
            print(f'Loaded data from {trajectory_path.name}')
            trajectories.append(trajectory)
        return trajectories, step_lookup

    def __len__(self):
        return len(self.step_lookup)

    def __getitem__(self, idx):
        trajectory_idx, step_idx = self.step_lookup[idx]
        sample = self.trajectories[trajectory_idx].get_item(
            step_idx, multiframe=self.multiframe)
        if self.transform:
            sample = self.transform(sample)
        return sample


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque([], maxlen=int(capacity))

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

    def push(self, obs, action, next_obs, done, reward):
        self.buffer.append((copy.deepcopy(obs), action.copy(),
                            copy.deepcopy(next_obs), done, reward))

    def sample(self, batch_size):
        replay_batch_size = min(batch_size, len(self.buffer))
        replay_batch = random.sample(self.buffer, replay_batch_size)
        return default_collate(replay_batch)


class MixedReplayBuffer(ReplayBuffer):
    '''
    Samples a fraction from the expert trajectories
    and the remainder from the replay buffer.
    '''

    def __init__(self,
                 expert_dataset,
                 capacity=1e6,
                 batch_size=64,
                 expert_sample_fraction=0.5):
        self.batch_size = batch_size
        self.expert_sample_fraction = expert_sample_fraction
        self.expert_batch_size = math.floor(batch_size * self.expert_sample_fraction)
        self.replay_batch_size = self.batch_size - self.expert_batch_size
        super().__init__(capacity)
        self.expert_dataset = expert_dataset
        self.expert_dataloader = self._initialize_dataloader()

    def _initialize_dataloader(self):
        return iter(DataLoader(self.expert_dataset,
                               shuffle=True,
                               batch_size=self.expert_batch_size,
                               num_workers=4,
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
