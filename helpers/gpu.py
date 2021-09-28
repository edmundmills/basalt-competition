from helpers.environment import ActionSpace

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


def expert_batch_to_device(batch):
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    states, actions, next_states, _done, _rewards = batch
    actions = ActionSpace.dataset_action_batch_to_actions(actions)
    mask = actions != -1
    actions = actions[mask]
    actions = th.from_numpy(actions).long().to(device)
    states = [state_component[mask] for state_component in states]
    next_states = [state_component[mask] for state_component in next_states]
    states, next_states = states_to_device((states, next_states), device)
    batch = states, actions, next_states, _done, _rewards
    return batch


def batch_to_device(batch):
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    states, actions, next_states, done, rewards = batch
    states, next_states = states_to_device((states, next_states), device)
    actions = actions.to(device)
    done = th.as_tensor(done).unsqueeze(1).float().to(device)
    rewards = rewards.float().unsqueeze(1).to(device)
    batch = states, actions, next_states, done, rewards
    return batch


def batches_to_device(expert_batch, replay_batch):
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    expert_states, expert_actions, expert_next_states, \
        expert_done, _expert_rewards = expert_batch
    replay_states, replay_actions, replay_next_states, \
        replay_done, _replay_rewards = replay_batch

    expert_actions = ActionSpace.dataset_action_batch_to_actions(expert_actions)
    expert_actions = th.from_numpy(expert_actions).unsqueeze(1)
    replay_actions = replay_actions.unsqueeze(1)
    expert_done = th.as_tensor(expert_done).float().to(device)
    replay_done = th.as_tensor(replay_done).float().to(device)

    mask = (expert_actions != -1).squeeze()
    expert_actions = expert_actions[mask].to(device)
    expert_states = [state_component[mask] for state_component in expert_states]
    expert_next_states = [state_component[mask]
                          for state_component in expert_next_states]

    expert_states, replay_states, expert_next_states, replay_next_states = \
        states_to_device((expert_states, replay_states,
                          expert_next_states, replay_next_states), device)

    expert_batch = expert_states, expert_actions, expert_next_states, \
        expert_done, _expert_rewards
    replay_batch = replay_states, replay_actions, replay_next_states, \
        replay_done, _replay_rewards
    return expert_batch, replay_batch
