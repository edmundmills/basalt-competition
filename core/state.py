from collections import namedtuple
from typing import Iterable, Tuple

import torch as th

State = namedtuple('State', 'spatial nonspatial hidden')
Transition = namedtuple('Transition', 'state action reward next_state done')
Sequence = namedtuple('Sequence', 'states actions rewards dones')


def cat_states(tuple_of_states: Iterable) -> Tuple[list, State]:
    """
    Concatenates a tuple of batches of states into a single batch of states.

    Returns the catted states along with a list of the original batch dimensions,
    in order to be able to split the states into their initial divisions later.
    """
    state_lengths = [states[0].size()[0] for states in tuple_of_states]
    all_states = State(*[th.cat(state_component, dim=0) for state_component
                         in zip(*tuple_of_states)])
    return all_states, state_lengths


def cat_transitions(tuple_of_transitions: Iterable) -> Transition:
    """Concatenates a tuple or list of transitions."""
    states, actions, rewards, next_states, dones = zip(*tuple_of_transitions)
    states, _ = cat_states(states)
    actions = th.cat(actions, dim=0)
    rewards = th.cat(rewards, dim=0)
    next_states, _ = cat_states(next_states)
    dones = th.cat(dones, dim=0)
    return Transition(states, actions, rewards, next_states, dones)


def sequence_to_transitions(sequence: Sequence) -> Transition:
    """
    Converts a sequence to a batch of transitions.
    
    The sequence stores all state information in a single sequence to reduce replay buffer
    RAM use. This method divides the state information of the sequence into current
    and next states. To include the final next state, the sequence state information is
    one longer than the rest of the elements.
    """
    states = sequence.states
    current_states = [state_component[:, :-1, ...] for state_component in states]
    next_states = [state_component[:, 1:, ...] for state_component in states]
    return Transition(State(*current_states),
                      sequence.actions,
                      sequence.rewards,
                      State(*next_states),
                      sequence.dones)


def update_hidden(state: State, hidden: th.Tensor):
    """Updates the hidden element of a given state with the given hidden value"""
    state = list(state)
    state[2] = hidden
    state = State(*state)
