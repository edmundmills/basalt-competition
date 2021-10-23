from core.state import State, Transition, Sequence cat_states, cat_transitions
from contexts.minerl.environment import MineRLContext

import torch as th


class GPULoader:
    def __init__(self, config):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        if config.context.name == 'MineRL':
            self.context = MineRLContext(config)
        self.load_sequences = config.lstm_layers > 0
        self.normalize_obs = config.normalize_obs
        means, stdevs = self.context.spatial_normalization
        means = means.reshape(3, 1, 1).tile(
            (config.n_observation_frames, 1, 1)).to(self.device)
        stdevs = stdevs.reshape(3, 1, 1).tile(
            (config.n_observation_frames, 1, 1)).to(self.device)
        self.spatial_normalization = means, stdevs
        self.mobilenet_normalization = (
            th.FloatTensor([0.485, 0.456, 0.406]).to(self.device).reshape(
                3, 1, 1).tile((config.n_observation_frames, 1, 1)),
            th.FloatTensor([0.229, 0.224, 0.225]).to(self.device).reshape(
                3, 1, 1).tile((config.n_observation_frames, 1, 1)))

    def normalize_state(self, state):
        state = list(state)
        if self.normalize_obs:
            state[0] = (state[0] - self.spatial_normalization[0]) \
                / self.spatial_normalization[1]
            state[0] = state[0] * self.mobilenet_normalization[1] \
                + self.mobilenet_normalization[0]
        else:
            state[0] /= 255.0
        state[1] /= self.nonspatial_normalization
        return State(*state)

    def states_from_sequence(self, sequence):
        states = sequence.states
        current_states = [state_component[:, :-1, ...] for state_component in states]
        next_states = [state_component[:, 1:, ...] for state_component in states]
        return State(*current_states), State(*next_states)

    def state_to_device(self, state):
        state = [state_component.unsqueeze(0).to(self.device, dtype=th.float)
                 for state_component in state]
        state = self.normalize_state(state)
        # add sequence dimension
        if self.load_sequences:
            state = [state_component.unsqueeze(0) for state_component in state]
        return State(*state)

    def states_to_device(self, tuple_of_states):
        # # this is slower, but may be better for larger batch sizes?
        # state_lengths = [states[0].size()[0] for states in tuple_of_states]
        # all_states = [th.cat(state_component, dim=0).to(device) for state_component
        #               in zip(*tuple_of_states)]
        # list_of_states = zip(*[th.split(state_component, state_lengths, dim=0)
        #                        for state_component in all_states])
        # return tuple(list_of_states)
        states = []
        for state in tuple_of_states:
            if len(state) != 0:
                state = [state_component.to(self.device, dtype=th.float)
                         for state_component in state]
                state = self.normalize_state(state)
                state = State(*state)
            states.append(state)
        return tuple(states)

    def transitions_to_device(self, transitions):
        if self.load_sequences:
            states, actions, rewards, dones = transitions
            next_states = []
        else:
            states, actions, rewards, next_states, dones = transitions
        states, next_states = self.states_to_device((states, next_states))
        if self.load_sequences:
            states, next_states = self.states_from_sequence(states)
        actions = actions.unsqueeze(-1).to(self.device)
        rewards = rewards.float().unsqueeze(-1).to(self.device)
        dones = th.as_tensor(dones).unsqueeze(-1).float().to(self.device)
        return Transition(states, actions, rewards, next_states, dones)
