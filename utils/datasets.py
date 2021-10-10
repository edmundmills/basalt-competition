from utils.trajectories import Trajectory
from utils.environment import ObservationSpace, ActionSpace

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
        self.lstm_hidden_size = config.lstm_hidden_size
        self.initial_hidden = th.zeros(self.lstm_hidden_size*2) \
            if self.lstm_hidden_size > 0 else None

        self.trajectories, self.step_lookup = self._load_data()
        print(f'Expert dataset initialized with {len(self.step_lookup)} steps')

    def _load_data(self):
        data = minerl.data.make(self.environment)
        trajectories = []
        step_lookup = []

        trajectory_paths = list(self.environment_path.iterdir())
        if self.environment == 'MineRLBasaltCreateVillageAnimalPen-v0':
            animal_pen_plains_path = \
                self.environment_path / 'MineRLBasaltCreateAnimalPenPlains-v0'
            trajectory_paths.extend(list(animal_pen_plains_path.iterdir()))
        trajectory_idx = 0
        for trajectory_path in trajectory_paths:
            if not trajectory_path.is_dir():
                continue
            if trajectory_path.name in [
                    'v3_villainous_black_eyed_peas_loch_ness_monster-2_95372-97535',
                    'MineRLBasaltCreateAnimalPenPlains-v0']:
                continue

            trajectory = Trajectory(n_observation_frames=self.n_observation_frames)
            step_idx = 0
            print(trajectory_path)
            for obs, action, _, _, done in data.load_data(str(trajectory_path)):
                trajectory.done = done
                action = ActionSpace.dataset_action_batch_to_actions(action)[0]
                if action == -1:
                    continue
                trajectory.append_obs(obs, self.initial_hidden)
                trajectory.actions.append(action)
                trajectory.rewards.append(0)
                step_lookup.append((trajectory_idx, step_idx))
                step_idx += 1
            print(f'Loaded data from {trajectory_path.name} ({step_idx} steps)')
            trajectories.append(trajectory)
            trajectory_idx += 1
            if self.debug_dataset and trajectory_idx >= 2:
                break
            if self.environment in ['MineRLTreechop-v0', 'MineRLNavigateDense-v0',
                                    'MineRLNavigateExtremeDense-v0'] \
                    and trajectory_idx >= 80:
                break
        return trajectories, step_lookup

    def __len__(self):
        return len(self.step_lookup)

    def __getitem__(self, idx):
        trajectory_idx, step_idx = self.step_lookup[idx]
        sample = self.trajectories[trajectory_idx][step_idx]
        return sample, idx


class TrajectorySegmentDataset(TrajectoryStepDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.segment_length = config.lstm_segment_length
        self.segment_lookup = self._identify_segments()
        print(f'Identified {len(self.segment_lookup)} sub-segments'
              f' of {self.segment_length} steps')
        self.curriculum_training = config.curriculum_training
        self.initial_curriculum_size = config.initial_curriculum_size
        if self.curriculum_training:
            self.update_curriculum(0)
            self.lookup = self.filtered_lookup
        else:
            self.lookup = self.segment_lookup

    def _identify_segments(self):
        segments = []
        for trajectory_idx, step_idx in self.step_lookup:
            if step_idx > self.segment_length + 1:
                segments.append((trajectory_idx, step_idx))
        return segments

    def __len__(self):
        return len(self.lookup)

    def __getitem__(self, idx):
        trajectory_idx, last_step_idx = self.lookup[idx]
        master_idx = self.cross_lookup[idx] if self.curriculum_training else idx
        sample = self.trajectories[trajectory_idx].get_segment(last_step_idx,
                                                               self.segment_length)
        return sample, master_idx

    def update_hidden(self, indices, hidden):
        for segment_idx, hidden in zip(indices.tolist(), hidden.unbind(dim=0)):
            trajectory_idx, step_idx = self.segment_lookup[segment_idx]
            self.trajectories[trajectory_idx].update_hidden(step_idx, hidden)

    def update_curriculum(self, curriculum_fraction):
        self.filtered_lookup, master_indices = \
            zip(*[[(t_idx, segment_idx), master_idx]
                  for master_idx, (t_idx, segment_idx) in enumerate(self.segment_lookup)
                  if (segment_idx <= len(self.trajectories[t_idx]) * curriculum_fraction
                      or segment_idx < self.initial_curriculum_size)])
        self.cross_lookup = {filtered_idx: master_idx
                             for filtered_idx, master_idx in enumerate(master_indices)}
        print(f'Expert curriculum updated, including {len(self.filtered_lookup)}'
              f' / {len(self.segment_lookup)} segments')
        self.lookup = self.filtered_lookup


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
        return sample, idx

    def current_trajectory(self):
        return self.trajectories[-1]

    def current_state(self):
        return self.current_trajectory().current_state()

    def new_trajectory(self):
        self.trajectories.append(Trajectory(
            n_observation_frames=self.n_observation_frames))

    def append_step(self, action, hidden, reward, next_obs, done):
        self.current_trajectory().actions.append(action)
        self.current_trajectory().rewards.append(reward)
        self.current_trajectory().append_obs(next_obs, hidden)
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
        assert(len(rewards) == len(self.step_lookup))
        for idx, (trajectory_idx, step_idx) in enumerate(self.step_lookup):
            self.trajectories[trajectory_idx].rewards[step_idx] = rewards[idx]
        print(f'{len(rewards)} replay steps labeled with rewards')

    def recent_frames(self, number_of_steps):
        total_steps = len(self.step_lookup)
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
        return sample, idx

    def increment_step(self):
        super().increment_step()
        if len(self.current_trajectory()) > self.segment_length + 1:
            self.segment_lookup.append(
                (len(self.trajectories) - 1, len(self.current_trajectory().actions) - 1))

    def sample(self, batch_size):
        replay_batch_size = min(batch_size, len(self.segment_lookup))
        sample_indices = random.sample(range(len(self.segment_lookup)), replay_batch_size)
        replay_batch = [self[idx] for idx in sample_indices]
        batch = default_collate(replay_batch)
        return batch

    def update_hidden(self, indices, hidden):
        for segment_idx, hidden in zip(indices.tolist(), hidden.unbind(dim=0)):
            trajectory_idx, step_idx = self.segment_lookup[segment_idx]
            self.trajectories[trajectory_idx].update_hidden(step_idx, hidden)
            _, _, next_state, _, _ = self.trajectories[trajectory_idx][step_idx]


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
        self.curriculum_training = config.curriculum_training
        self.curriculum_refresh_steps = config.curriculum_refresh_steps
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
            batch = next(self.expert_dataloader)
        except StopIteration:
            if self.curriculum_training:
                self.expert_dataset.update_curriculum(self.curriculum_fraction)
            self.expert_dataloader = self._initialize_dataloader()
            batch = next(self.expert_dataloader)
        return batch

    def sample(self, batch_size, include_idx=False):
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

    def update_curriculum(self, step, curriculum_fraction):
        self.curriculum_fraction = curriculum_fraction
        current_curriculum_inclusion = len(self.expert_dataset.filtered_lookup) / \
            len(self.expert_dataset.segment_lookup)
        if step % self.curriculum_refresh_steps == 0 and current_curriculum_inclusion < 1:
            self.expert_dataset.update_curriculum(self.curriculum_fraction)
            self.expert_dataloader = self._initialize_dataloader()
        curriculum_inclusion = len(self.expert_dataset.filtered_lookup) / \
            len(self.expert_dataset.segment_lookup)
        return curriculum_inclusion


class MixedSegmentReplayBuffer(MixedReplayBuffer, SegmentReplayBuffer):
    def __init__(self, expert_dataset, config,
                 batch_size, initial_replay_buffer=None):
        super().__init__(expert_dataset, config, batch_size, initial_replay_buffer)
        if initial_replay_buffer is not None:
            self.segment_lookup = initial_replay_buffer.segment_lookup

    def update_hidden(self, replay_indices, replay_hidden, expert_indices, expert_hidden):
        super().update_hidden(replay_indices, replay_hidden)
        self.expert_dataset.update_hidden(expert_indices, expert_hidden)
