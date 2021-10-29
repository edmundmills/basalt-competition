from core.trajectories import Trajectory
from core.trajectory_viewer import TrajectoryViewer
from contexts.minerl.dataset import MineRLDatasetBuilder

from collections import deque

import torch as th
import math
import random

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


class TrajectoryStepDataset(Dataset):
    def __init__(self, config, debug_dataset=False):
        if config.context.name == 'MineRL':
            dataset_builder = MineRLDatasetBuilder(config, debug_dataset)
        self.trajectories, self.step_lookup, self.stats = dataset_builder.load_data()
        if 'entropy' in self.stats.keys():
            self.expert_policy_entropy = self.stats['entropy']
        self.master_lookup = self.step_lookup
        self.active_lookup = self.step_lookup
        # to recover step index when using a different lookup array when sampling
        self.cross_lookup = None
        print(f'Expert dataset initialized with {len(self.step_lookup)} steps')

    def __len__(self):
        return len(self.active_lookup)

    def __getitem__(self, idx):
        trajectory_idx, step_idx = self.active_lookup[idx]
        sample = self.trajectories[trajectory_idx][step_idx]
        master_idx = self.cross_lookup[idx] if self.cross_lookup is not None else idx
        return sample, master_idx


class TrajectorySequenceDataset(TrajectoryStepDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.sequence_length = config.model.lstm_sequence_length
        self.sequence_lookup = self._identify_sequences()
        self.master_lookup = self.sequence_lookup
        self.active_lookup = self.sequence_lookup
        print(f'Identified {len(self.sequence_lookup)} sub-sequences'
              f' of {self.sequence_length} steps')

    def _identify_sequences(self):
        sequences = []
        for trajectory_idx, step_idx in self.step_lookup:
            if step_idx >= self.sequence_length - 1:
                sequences.append((trajectory_idx, step_idx))
        return sequences

    def __len__(self):
        return len(self.active_lookup)

    def __getitem__(self, idx):
        trajectory_idx, last_step_idx = self.active_lookup[idx]
        master_idx = self.cross_lookup[idx] if self.cross_lookup is not None else idx
        sample = self.trajectories[trajectory_idx].get_sequence(last_step_idx,
                                                                self.sequence_length)
        return sample, master_idx

    def update_hidden(self, indices, hidden):
        for sequence_idx, hidden in zip(indices.tolist(), hidden.unbind(dim=0)):
            trajectory_idx, step_idx = self.sequence_lookup[sequence_idx]
            self.trajectories[trajectory_idx].update_hidden(step_idx, hidden)


class ReplayBuffer:
    def __init__(self, config, initial_replay_buffer=None):
        self.trajectories = [Trajectory()]
        self.step_lookup = []
        if initial_replay_buffer is not None:
            self.trajectories = initial_replay_buffer.trajectories
            self.step_lookup = initial_replay_buffer.step_lookup

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
        self.trajectories.append(Trajectory())

    def append_step(self, action, reward, next_state, done, **kwargs):
        self.current_trajectory().append_step(action, reward, next_state, done, **kwargs)
        self.increment_step()

    def increment_step(self):
        self.step_lookup.append(
            (len(self.trajectories) - 1, len(self.current_trajectory().actions) - 1))

    def sample(self, batch_size):
        replay_batch_size = min(batch_size, len(self.step_lookup))
        sample_indices = random.sample(range(len(self.step_lookup)), replay_batch_size)
        print(self.step_lookup)
        print(sample_indices)
        replay_batch = [self[idx] for idx in sample_indices]
        batch = default_collate(replay_batch)
        return batch

    def recent_frames(self, number_of_steps):
        return TrajectoryViewer.dataset_recent_frames(self, number_of_steps)


class SequenceReplayBuffer(ReplayBuffer):
    def __init__(self, config, initial_replay_buffer=None):
        super().__init__(config, initial_replay_buffer)
        self.sequence_lookup = []
        self.sequence_length = config.model.lstm_sequence_length
        if initial_replay_buffer is not None:
            self.sequence_lookup = initial_replay_buffer.sequence_lookup

    def __len__(self):
        return len(self.sequence_lookup)

    def __getitem__(self, idx):
        trajectory_idx, sequence_idx = self.sequence_lookup[idx]
        sample = self.trajectories[trajectory_idx].get_sequence(sequence_idx,
                                                                self.sequence_length)
        return sample, idx

    def increment_step(self):
        super().increment_step()
        if len(self.current_trajectory()) > self.sequence_length + 1:
            self.sequence_lookup.append(
                (len(self.trajectories) - 1, len(self.current_trajectory().actions) - 1))

    def sample(self, batch_size):
        replay_batch_size = min(batch_size, len(self.sequence_lookup))
        sample_indices = random.sample(
            range(len(self.sequence_lookup)), replay_batch_size)
        replay_batch = [self[idx] for idx in sample_indices]
        batch = default_collate(replay_batch)
        return batch

    def update_hidden(self, indices, hidden):
        for sequence_idx, hidden in zip(indices.tolist(), hidden.unbind(dim=0)):
            trajectory_idx, step_idx = self.sequence_lookup[sequence_idx]
            self.trajectories[trajectory_idx].update_hidden(step_idx, hidden)
            _, _, next_state, _, _ = self.trajectories[trajectory_idx][step_idx]


class MixedReplayBuffer(ReplayBuffer):
    '''
    Samples a fraction from the expert trajectories
    and the remainder from the replay buffer.
    '''

    def __init__(self, expert_dataset, config,
                 batch_size, initial_replay_buffer=None):
        super().__init__(config, initial_replay_buffer)
        self.batch_size = batch_size
        self.expert_sample_fraction = config.method.expert_sample_fraction
        self.expert_batch_size = math.floor(batch_size * self.expert_sample_fraction)
        self.replay_batch_size = self.batch_size - self.expert_batch_size
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
            batch = next(self.expert_dataloader)
        except StopIteration:
            self.expert_dataloader = self._initialize_dataloader()
            batch = next(self.expert_dataloader)
        return batch

    def sample(self, batch_size):
        return self.sample_expert(), self.sample_replay()


class MixedSequenceReplayBuffer(MixedReplayBuffer, SequenceReplayBuffer):
    def __init__(self, expert_dataset, config,
                 batch_size, initial_replay_buffer=None):
        super().__init__(expert_dataset, config, batch_size, initial_replay_buffer)
        if initial_replay_buffer is not None:
            self.sequence_lookup = initial_replay_buffer.sequence_lookup

    def update_hidden(self, replay_indices, replay_hidden, expert_indices, expert_hidden):
        super().update_hidden(replay_indices, replay_hidden)
        self.expert_dataset.update_hidden(expert_indices, expert_hidden)
