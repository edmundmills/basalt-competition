from core.state import State, Transition, sequence_to_transitions
from contexts.minerl.environment import MineRLContext

from typing import Iterable, NamedTuple, Tuple

import torch as th


class GPULoader:
    """
    Handles loading of tensors onto the gpu.

    Initialized with normalization tensors to avoid recreating them on the gpu.

    Can load:
    Individual state: state_to_device
    Batch of states: states_to_device
    Batch of transitions: transitions_to_device
    Batch of sequences: transitions_to_device
    """
    def __init__(self, config):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        if config.context.name == 'MineRL':
            context = MineRLContext(config)
        self.load_sequences = config.model.lstm_layers > 0
        self.normalize_obs = config.context.normalize_obs
        means, stdevs = context.spatial_normalization
        means = means.reshape(3, 1, 1).tile(
            (config.model.n_observation_frames, 1, 1)).to(self.device)
        stdevs = stdevs.reshape(3, 1, 1).tile(
            (config.model.n_observation_frames, 1, 1)).to(self.device)
        self.spatial_normalization = means, stdevs
        self.nonspatial_normalization = context.nonspatial_normalization.to(self.device)
        self.mobilenet_normalization = (
            th.FloatTensor([0.485, 0.456, 0.406]).to(self.device).reshape(
                3, 1, 1).tile((config.model.n_observation_frames, 1, 1)),
            th.FloatTensor([0.229, 0.224, 0.225]).to(self.device).reshape(
                3, 1, 1).tile((config.model.n_observation_frames, 1, 1)))

    def normalize_state(self, state: State) -> State:
        """Normalizes a state on the gpu according to config"""
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

    def state_to_device(self, state: State) -> State:
        """Loads the multiple components of a state to the gpu."""
        state = [state_component.unsqueeze(0).to(self.device, dtype=th.float)
                 for state_component in state]
        # add sequence dimension
        if self.load_sequences:
            state = [state_component.unsqueeze(0) for state_component in state]
        state = State(*state)
        state = self.normalize_state(state)
        return state

    def states_to_device(self, tuple_of_states: Iterable) -> Tuple:
        """Loads a tuple of states or batches of states onto the gpu."""
        states = []
        for state in tuple_of_states:
            if len(state) != 0:
                state = [state_component.to(self.device, dtype=th.float)
                         for state_component in state]
                state = State(*state)
                state = self.normalize_state(state)
            states.append(state)
        return tuple(states)

    def transitions_to_device(self, transitions: NamedTuple) -> Transition:
        """
        Loads a batch of transitions or sequences onto the gpu.
        
        If loading sequences, converts the sequence into transitions.
        """
        if self.load_sequences:
            transitions = sequence_to_transitions(transitions)
        states, actions, rewards, next_states, dones = transitions
        states, next_states = self.states_to_device((states, next_states))
        actions = actions.unsqueeze(-1).long().to(self.device)
        rewards = th.as_tensor(rewards).unsqueeze(-1).float().to(self.device)
        dones = th.as_tensor(dones).unsqueeze(-1).long().to(self.device)
        return Transition(states, actions, rewards, next_states, dones)
