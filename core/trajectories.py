from core.state import State, Transition, Sequence
from core.trajectory_viewer import TrajectoryViewer

import torch as th


class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.done = False
        self.additional_step_data = {}

    def __len__(self):
        return max(0, len(self.states) - 1)

    def __getitem__(self, idx):
        is_last_step = idx + 1 == len(self)
        done = is_last_step and self.done
        next_state = self.states[idx + 1]
        reward = self.rewards[idx]
        return Transition(self.states[idx], self.actions[idx], reward, next_state, done)

    def current_state(self):
        return self.states[-1]

    def update_hidden(self, idx, new_hidden):
        is_last_step = idx + 1 == len(self)
        done = is_last_step and self.done
        if not done:
            idx += 1
        next_state = self.states[idx]
        *state_components, hidden = next_state
        self.states[idx] = State(*state_components, new_hidden)

    def get_sequence(self, last_step_idx, sequence_length):
        is_last_step = last_step_idx + 1 == len(self)
        dones = th.zeros(sequence_length)
        dones[-1] = 1 if is_last_step and self.done else 0
        actions = th.LongTensor(
            self.actions[last_step_idx - sequence_length:last_step_idx])
        states = self.states[last_step_idx - sequence_length:last_step_idx + 1]
        states = State(*[th.stack(state_component) for state_component in zip(*states)])
        rewards = th.FloatTensor(
            self.rewards[last_step_idx - sequence_length:last_step_idx])
        return Sequence(states, actions, rewards, dones)

    def append_step(self, action, reward, next_state, done, **kwargs):
        self.actions.append(action)
        self.rewards.append(reward)
        self.states.append(next_state)
        self.done = done
        for k, v in kwargs.items():
            if k not in self.additional_step_data.keys():
                self.additional_step_data[k] = []
            self.additional_step_data[k].append(v)

    def save_video(self, save_dir_path, filename):
        return TrajectoryViewer(self).to_video(save_dir_path, filename)

    def view(self):
        return TrajectoryViewer(self).view()
