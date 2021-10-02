from helpers.environment import ActionSpace, ObservationSpace

import torch as th


def states_to_device(tuple_of_states, device):
    # # this is slower, but may be better for larger batch sizes?
    # state_lengths = [states[0].size()[0] for states in tuple_of_states]
    # all_states = [th.cat(state_component, dim=0).to(device) for state_component
    #               in zip(*tuple_of_states)]
    # list_of_states = zip(*[th.split(state_component, state_lengths, dim=0)
    #                        for state_component in all_states])
    # return tuple(list_of_states)
    states = []
    for state in tuple_of_states:
        state = [state_component.to(device, dtype=th.float) for state_component in state]
        state = ObservationSpace.normalize_state(state)
        states.append(state)
    return tuple(states)


def cat_states(tuple_of_states):
    state_lengths = [states[0].size()[0] for states in tuple_of_states]
    all_states = [th.cat(state_component, dim=0) for state_component
                  in zip(*tuple_of_states)]
    return all_states, state_lengths


def cat_batches(tuple_of_batches):
    states, actions, next_states, done, reward = zip(*tuple_of_batches)
    states, _ = cat_states(states)
    actions = th.cat(actions, dim=0)
    next_states, _ = cat_states(next_states)
    done = th.cat(done, dim=0)
    reward = th.cat(reward, dim=0)
    batch = states, actions, next_states, done, reward
    return batch


def disable_gradients(network):
    for param in network.parameters():
        param.requires_grad = False


def expert_batch_to_device(batch):
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    states, actions, next_states, done, rewards = batch
    mask = actions != -1
    actions = actions[mask]
    states = [state_component[mask] for state_component in states]
    next_states = [state_component[mask] for state_component in next_states]
    done = done[mask]
    states, next_states = states_to_device((states, next_states), device)
    actions = actions.unsqueeze(1).to(device)
    done = th.as_tensor(done).unsqueeze(1).float().to(device)
    rewards = rewards.float().unsqueeze(1).to(device)
    batch = states, actions, next_states, done, rewards
    return batch


def batch_to_device(batch):
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    states, actions, next_states, done, rewards = batch
    states, next_states = states_to_device((states, next_states), device)
    actions = actions.unsqueeze(1).to(device)
    done = th.as_tensor(done).unsqueeze(1).float().to(device)
    rewards = rewards.float().unsqueeze(1).to(device)
    batch = states, actions, next_states, done, rewards
    return batch


def batches_to_device(expert_batch, replay_batch):
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    expert_states, expert_actions, expert_next_states, \
        expert_done, _expert_rewards = expert_batch
    replay_states, replay_actions, replay_next_states, \
        replay_done, replay_rewards = replay_batch

    expert_actions = expert_actions.unsqueeze(1)

    mask = (expert_actions != -1).squeeze()
    expert_actions = expert_actions[mask].to(device)
    replay_actions = replay_actions.unsqueeze(1).to(device)
    expert_states = [state_component[mask] for state_component in expert_states]
    expert_next_states = [state_component[mask]
                          for state_component in expert_next_states]

    expert_states, replay_states, expert_next_states, replay_next_states = \
        states_to_device((expert_states, replay_states,
                          expert_next_states, replay_next_states), device)

    expert_done = th.as_tensor(expert_done[mask]).float().to(device).unsqueeze(1)
    replay_done = th.as_tensor(replay_done).float().to(device).unsqueeze(1)
    replay_rewards = replay_rewards.float().unsqueeze(1).to(device)

    expert_batch = expert_states, expert_actions, expert_next_states, \
        expert_done, _expert_rewards
    replay_batch = replay_states, replay_actions, replay_next_states, \
        replay_done, replay_rewards
    return expert_batch, replay_batch
