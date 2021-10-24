from core.trajectories import Trajectory
from contexts.minerl.dataset import MineRLDatasetBuilder

from collections import deque

import torch as th
import math
import random

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


class TrajectoryStepDataset(Dataset):
    def __init__(self, config, debug_dataset=False):
        self.n_observation_frames = config.n_observation_frames
        self.debug_dataset = debug_dataset
        self.lstm_hidden_size = config.lstm_hidden_size
        self.initial_hidden = th.zeros(self.lstm_hidden_size*2) \
            if self.lstm_hidden_size > 0 else None
        if config.context.name == 'MineRL':
            dataset_builder = MineRLDatasetBuilder(config)
        self.trajectories, self.step_lookup = dataset_builder.load_data()
        print(f'Expert dataset initialized with {len(self.step_lookup)} steps')

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
        self.emphasize_new_samples = config.emphasize_new_samples
        self.emphasized_fraction = config.emphasized_fraction
        self.extracurricular_sparsity = config.extracurricular_sparsity
        self.emphasis_relative_sample_frequency = \
            config.emphasis_relative_sample_frequency
        self.final_curriculum_fraction = 1 + self.emphasized_fraction \
            if self.emphasize_new_samples else 1

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
        random_seed = random.randint(0, self.extracurricular_sparsity - 1)
        self.filtered_lookup, master_indices = \
            zip(*[[(t_idx, segment_idx), master_idx]
                  for master_idx, (t_idx, segment_idx) in enumerate(self.segment_lookup)
                  if (segment_idx <= (len(self.trajectories[t_idx]) * curriculum_fraction)
                      or segment_idx < self.initial_curriculum_size
                      or (segment_idx + random_seed) % self.extracurricular_sparsity == 0)
                  ])
        self.filtered_lookup = list(self.filtered_lookup)
        master_indices = list(master_indices)
        self.current_curriculum_length = len(self.filtered_lookup)
        if self.emphasize_new_samples and curriculum_fraction > self.emphasized_fraction \
                and curriculum_fraction < self.final_curriculum_fraction:
            for i in range(self.emphasis_relative_sample_frequency - 1):
                emphasis_lookup, emphasis_master_indices = \
                    zip(*[[(t_idx, segment_idx), master_idx]
                          for master_idx, (t_idx, segment_idx)
                          in enumerate(self.segment_lookup)
                          if (segment_idx <=
                              (len(self.trajectories[t_idx]) * curriculum_fraction)
                              and segment_idx > (len(self.trajectories[t_idx])
                                                 * (curriculum_fraction
                                                    - self.emphasized_fraction)))])
                self.filtered_lookup.extend(emphasis_lookup)
                print(f'{len(emphasis_lookup)} samples emphasized')
                master_indices.extend(emphasis_master_indices)
        self.cross_lookup = {filtered_idx: master_idx
                             for filtered_idx, master_idx in enumerate(master_indices)}
        print(f'Expert curriculum updated, including {self.current_curriculum_length}'
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

    def append_step(self, action, reward, next_state, done):
        self.current_trajectory().actions.append(action)
        self.current_trajectory().rewards.append(reward)
        self.current_trajectory().states.append(next_state)
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

    def update_curriculum(self, step, curriculum_fraction):
        self.curriculum_fraction = curriculum_fraction
        current_curriculum_inclusion = self.expert_dataset.current_curriculum_length / \
            len(self.expert_dataset.segment_lookup)
        if step % self.curriculum_refresh_steps == 0 and self.curriculum_fraction \
                < self.expert_dataset.final_curriculum_fraction:
            self.expert_dataset.update_curriculum(self.curriculum_fraction)
            self.expert_dataloader = self._initialize_dataloader()
        curriculum_inclusion = self.expert_dataset.current_curriculum_length / \
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
