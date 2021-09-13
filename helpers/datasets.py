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
                 n_observation_frames=1,
                 debug_dataset=False):
        self.n_observation_frames = n_observation_frames
        self.debug_dataset = debug_dataset
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
        trajectory_idx = 0
        for trajectory_path in trajectory_paths:
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
            trajectory_idx += 1
            if self.debug_dataset and trajectory_idx > 2:
                break
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
    def __init__(self, n_observation_frames=1, reward=True):
        self.n_observation_frames = n_observation_frames
        self.trajectories = [Trajectory()]
        self.step_lookup = []
        self.reward = reward

    def __len__(self):
        return len(self.step_lookup)

    def __getitem__(self, idx):
        trajectory_idx, step_idx = step_lookup[idx]
        return self.trajectories[trajectory_idx].get_item(
            step_idx, n_observation_frames=self.n_observation_frames, reward=self.reward)

    def current_trajectory(self):
        return self.trajectories[-1]

    def increment_step(self):
        self.step_lookup.append(
            (len(self.trajectories), len(self.current_trajectory().actions)))

    def add_trajectory(self):
        self.trajectories.append(Trajectory())

    def sample(self, batch_size):
        replay_batch_size = min(batch_size, len(self.buffer))
        replay_batch = random.sample(self, replay_batch_size)
        return default_collate(replay_batch)


class MixedReplayBuffer(ReplayBuffer):
    '''
    Samples a fraction from the expert trajectories
    and the remainder from the replay buffer.
    '''

    def __init__(self,
                 expert_dataset,
                 batch_size=64,
                 expert_sample_fraction=0.5,
                 n_observation_frames=1):
        self.batch_size = batch_size
        self.expert_sample_fraction = expert_sample_fraction
        self.expert_batch_size = math.floor(batch_size * self.expert_sample_fraction)
        self.replay_batch_size = self.batch_size - self.expert_batch_size
        super().__init__(n_observation_frames=n_observation_frames)
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
