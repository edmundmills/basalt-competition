import torch as th


def states_to_device(tuple_of_states, device):
    all_states = [th.cat(state_component, dim=0).to(device) for state_component
                  in zip(*tuple_of_states)]
    list_of_states = zip(*[th.chunk(state_component, 2, dim=0)
                           for state_component in all_states])
    return tuple(list_of_states)


def disable_gradients(network):
    for param in network.parameters():
        param.requires_grad = False
