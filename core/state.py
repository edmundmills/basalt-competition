from collections import namedtuple

State = namedtuple('State', 'spatial nonspatial hidden')
Transition = namedtuple('Transition', 'state action reward next_state done')
Sequence = namedtuple('Sequence', 'state action reward done')


def cat_states(tuple_of_states):
    state_lengths = [states[0].size()[0] for states in tuple_of_states]
    all_states = State(*[th.cat(state_component, dim=0) for state_component
                         in zip(*tuple_of_states)])
    return all_states, state_lengths


def cat_transitions(tuple_of_transitions):
    states, actions, rewards, next_states, dones = zip(*tuple_of_transitions)
    states, _ = cat_states(states)
    actions = th.cat(actions, dim=0)
    rewards = th.cat(rewards, dim=0)
    next_states, _ = cat_states(next_states)
    dones = th.cat(dones, dim=0)
    return Transition(states, actions, rewards, next_states, dones)
