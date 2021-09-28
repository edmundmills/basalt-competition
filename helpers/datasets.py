from helpers.trajectories import Trajectory
from helpers.environment import ObservationSpace

import minerl

from pathlib import Path
import os

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
            if self.debug_dataset and trajectory_idx >= 2:
                break
            if self.environment == 'MineRLTreechop-v0' and trajectory_idx >= 80:
                break
        return trajectories, step_lookup

    def __len__(self):
        return len(self.step_lookup)

    def __getitem__(self, idx):
        trajectory_idx, step_idx = self.step_lookup[idx]
        sample = self.trajectories[trajectory_idx].get_item(
            step_idx, n_observation_frames=self.n_observation_frames)
        return sample


class ReplayBuffer:
    def __init__(self, config):
        self.n_observation_frames = config.n_observation_frames
        self.trajectories = [Trajectory()]
        self.step_lookup = []

    def __len__(self):
        return len(self.step_lookup)

    def __getitem__(self, idx):
        trajectory_idx, step_idx = self.step_lookup[idx]
        sample = self.trajectories[trajectory_idx].get_item(
            step_idx, n_observation_frames=self.n_observation_frames,
            reward=True)
        return sample

    def current_trajectory(self):
        return self.trajectories[-1]

    def current_state(self):
        return self.current_trajectory().current_state(
            n_observation_frames=self.n_observation_frames)

    def new_trajectory(self):
        self.trajectories.append(Trajectory())

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
        batch = ObservationSpace.batch_obs_to_states(batch)
        return batch

    def recent_frames(self, number_of_steps):
        self._n_observation_frames = self.n_observation_frames
        self.n_observation_frames = 1
        total_steps = len(self)
        steps = min(number_of_steps, total_steps)
        frame_skip = 2
        frames = int(round(total_steps / (frame_skip + 1)))
        step_rate = 20  # steps / second
        frame_rate = int(round(step_rate / (frame_skip + 1)))
        step_indices = [total_steps - steps + frame *
                        (frame_skip + 1) for frame in range(frames)]
        images = [(self[min(step_idx, total_steps-1)][0][0][0:3, :, :].numpy()
                   * 255).astype(np.uint8)
                  for step_idx in step_indices]
        images = np.stack(images, 0)
        self.n_observation_frames = self._n_observation_frames
        return images, frame_rate


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
        sample = ObservationSpace.batch_obs_to_states(sample)
        return sample

    def sample(self, batch_size):
        return self.sample_expert(), self.sample_replay()
