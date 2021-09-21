import torch as th


def states_to_device(tuple_of_states, device):
    state_lengths = [states[0].size()[0] for states in tuple_of_states]
    all_states = [th.cat(state_component, dim=0).to(device) for state_component
                  in zip(*tuple_of_states)]
    list_of_states = zip(*[th.split(state_component, state_lengths, dim=0)
                           for state_component in all_states])
    return tuple(list_of_states)


def cat_states(tuple_of_states):
    state_lengths = [states[0].size()[0] for states in tuple_of_states]
    all_states = [th.cat(state_component, dim=0) for state_component
                  in zip(*tuple_of_states)]
    return all_states, state_lengths


def disable_gradients(network):
    for param in network.parameters():
        param.requires_grad = False
