from helpers.trajectories import Trajectory
from helpers.environment import ObservationSpace, ActionSpace

import minerl

from pathlib import Path
import os
import time
from collections import deque

import torch as th
import math
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


class TrajectoryStepDataset(Dataset):
    def __init__(self, config, debug_dataset=False):
        self.n_observation_frames = config.n_observation_frames
        self.debug_dataset = debug_dataset
        self.data_root = Path(os.getenv('MINERL_DATA_ROOT'))
        self.environment = os.getenv('MINERL_ENVIRONMENT')
        self.environment_path = self.data_root / self.environment
        self.trajectories, self.step_lookup = self._load_data()
        print(f'Expert dataset initialized with {len(self)} steps')

    def _load_data(self):
        data = minerl.data.make(self.environment)
        trajectories = []
        step_lookup = []

        trajectory_paths = self.environment_path.iterdir()
        trajectory_idx = 0
        for trajectory_path in trajectory_paths:
            if not trajectory_path.is_dir():
                continue

            trajectory = Trajectory(n_observation_frames=self.n_observation_frames)
            step_idx = 0
            noops = deque([True, True], maxlen=2)
            for obs, action, _, _, done in data.load_data(str(trajectory_path)):
                trajectory.done = done
                action = ActionSpace.dataset_action_batch_to_actions(action)[0]
                noops.append(action == -1)
                # skip adding states that are not part of a non-no-op transition
                if noops[0] and noops[1]:
                    continue
                trajectory.append_obs(obs)
                trajectory.actions.append(action)
                trajectory.rewards.append(0)
                step_lookup.append((trajectory_idx, step_idx))
                step_idx += 1
            print(f'Loaded data from {trajectory_path.name} ({step_idx} steps)')
            trajectories.append(trajectory)
            trajectory_idx += 1
            if self.debug_dataset and trajectory_idx >= 2:
                break
            if self.environment == 'MineRLTreechop-v0' and trajectory_idx >= 80:
                break
        return trajectories, step_lookup

    def __len__(self):
        return len(self.step_lookup)

    def __getitem__(self, idx):
        trajectory_idx, step_idx = self.step_lookup[idx]
        sample = self.trajectories[trajectory_idx][step_idx]
        return sample


class TrajectorySegmentDataset(TrajectoryStepDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.segment_length = config.lstm_segment_length
        self.segment_lookup = self._identify_segments()

    def _identify_segments():
        segments = []
        for trajectory_idx, step_idx in self.step_lookup:
            if step_idx > self.segment_length:
                segments.append((trajectory_idx, step_idx))
        return segments

    def __len__(self):
        return len(self.segment_lookup)

    def __getitem__(self, idx):
        trajectory_idx, last_step_idx = self.segment_lookup[idx]
        sample = self.trajectories[trajectory_idx].get_segment(last_step_idx,
                                                               self.segment_length)
        return sample


class ReplayBuffer:
    def __init__(self, config):
        self.n_observation_frames = config.n_observation_frames
        self.trajectories = [Trajectory(n_observation_frames=self.n_observation_frames)]
        self.step_lookup = []

    def __len__(self):
        return len(self.step_lookup)

    def __getitem__(self, idx):
        trajectory_idx, step_idx = self.step_lookup[idx]
        sample = self.trajectories[trajectory_idx][step_idx]
        return sample

    def current_trajectory(self):
        return self.trajectories[-1]

    def current_state(self):
        return self.current_trajectory().current_state()

    def new_trajectory(self):
        self.trajectories.append(Trajectory(
            n_observation_frames=self.n_observation_frames))

    def append_step(self, action, reward, next_obs, done):
        self.current_trajectory().actions.append(action)
        self.current_trajectory().rewards.append(reward)
        self.current_trajectory().append_obs(next_obs)
        self.current_trajectory().done = done
        self.increment_step()

    def increment_step(self):
        self.step_lookup.append(
            (len(self.trajectories) - 1, len(self.current_trajectory().actions) - 1))

    def sample(self, batch_size):
        replay_batch_size = min(batch_size, len(self.step_lookup))
        sample_indices = random.sample(range(len(self.step_lookup)), replay_batch_size)
        replay_batch = [self[idx] for idx in sample_indices]
        batch = default_collate(replay_batch)
        return batch

    def update_rewards(self, rewards):
        assert(len(rewards) == len(self))
        for idx, (trajectory_idx, step_idx) in enumerate(self.step_lookup):
            self.trajectories[trajectory_idx].rewards[step_idx] = rewards[idx]
        print(f'{len(rewards)} replay steps labeled with rewards')

    def recent_frames(self, number_of_steps):
        total_steps = len(self)
        steps = min(number_of_steps, total_steps)
        frame_skip = 2
        frames = int(round(total_steps / (frame_skip + 1)))
        step_rate = 20  # steps / second
        frame_rate = int(round(step_rate / (frame_skip + 1)))
        step_indices = [min(total_steps - steps + frame * (frame_skip + 1),
                            total_steps - 1)
                        for frame in range(frames)]
        indices = [self.step_lookup[step_index] for step_index in step_indices]
        images = [self.trajectories[trajectory_idx].get_pov(step_idx)
                  for trajectory_idx, step_idx in indices]
        images = [(image.numpy()).astype(np.uint8)
                  for image in images]
        images = np.stack(images, 0)
        return images, frame_rate


class SegmentReplayBuffer(ReplayBuffer):
    def __init__(self, config):
        super().__init__(config)
        self.segment_lookup = []
        self.segment_length = config.lstm_segment_length

    def __len__(self):
        return len(self.segment_lookup)

    def __getitem__(self, idx):
        trajectory_idx, segment_idx = self.segment_lookup[idx]
        sample = self.trajectories[trajectory_idx].get_segment(segment_idx,
                                                               self.segment_length)
        return sample

    def increment_step(self):
        super().increment_step()
        if len(self.current_trajectory()) > self.segment_length:
            self.segment_lookup.append(
                (len(self.trajectories) - 1, len(self.current_trajectory().actions) - 1))

    def sample(self, batch_size):
        replay_batch_size = min(batch_size, len(self.segment_lookup))
        sample_indices = random.sample(range(len(self.segment_lookup)), replay_batch_size)
        replay_batch = [self[idx] for idx in sample_indices]
        batch = default_collate(replay_batch)
        return batch


class MixedReplayBuffer(ReplayBuffer):
    '''
    Samples a fraction from the expert trajectories
    and the remainder from the replay buffer.
    '''

    def __init__(self, expert_dataset, config,
                 batch_size, initial_replay_buffer=None):
        self.batch_size = batch_size
        self.expert_sample_fraction = config.method.expert_sample_fraction
        self.expert_batch_size = math.floor(batch_size * self.expert_sample_fraction)
        self.replay_batch_size = self.batch_size - self.expert_batch_size
        super().__init__(config)
        if initial_replay_buffer is not None:
            self.trajectories = initial_replay_buffer.trajectories
            self.step_lookup = initial_replay_buffer.step_lookup
        self.expert_dataset = expert_dataset
        self.expert_dataloader = self._initialize_dataloader()

    def _initialize_dataloader(self):
        return iter(DataLoader(self.expert_dataset,
                               shuffle=True,
                               batch_size=self.expert_batch_size,
                               num_workers=4,
                               drop_last=True))

    def sample_replay(self):
        return super().sample(self.replay_batch_size)

    def sample_expert(self):
        try:
            sample = next(self.expert_dataloader)
        except StopIteration:
            self.expert_dataloader = self._initialize_dataloader()
            sample = next(self.expert_dataloader)
        return sample

    def sample(self, batch_size):
        return self.sample_expert(), self.sample_replay()

    def update_rewards(self, replay_rewards, expert_rewards):
        super().update_rewards(replay_rewards)
        rewards_idx = 0
        for _, (trajectory_idx, step_idx) in enumerate(self.expert_dataset.step_lookup):
            if self.expert_dataset.trajectories[trajectory_idx].actions[step_idx] == -1:
                continue
            self.expert_dataset.trajectories[trajectory_idx].rewards[step_idx] = \
                expert_rewards[rewards_idx]
            rewards_idx += 1
        print(f'{len(expert_rewards)} expert steps labeled with rewards')


class MixedSegmentReplayBuffer(MixedReplayBuffer, SegmentReplayBuffer):
    def __init__(self, expert_dataset, config,
                 batch_size, initial_replay_buffer=None):
        super().__init__(expert_dataset, config, batch_size, initial_replay_buffer)
        if initial_replay_buffer is not None:
            self.segment_lookup = initial_replay_buffer.segment_lookup
