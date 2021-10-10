from utils.environment import ActionSpace, ObservationSpace

import torch as th


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
    return states, actions, next_states, done, reward


def disable_gradients(network):
    for param in network.parameters():
        param.requires_grad = False


class GPULoader:
    def __init__(self, config):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        starting_count = th.FloatTensor(
            list(ObservationSpace.starting_inventory().values())).reshape(1, -1)
        ones = th.ones(starting_count.size())
        self.item_normalization = th.cat((starting_count, ones), dim=1).to(self.device)
        self.loading_sequences = config.lstm_layers > 0
        self.normalize_obs = config.normalize_obs
        self.env = config.env.name
        self.mobilenet_normalization = (
            th.FloatTensor([0.485, 0.456, 0.406]).to(self.device).reshape(
                3, 1, 1).tile((config.n_observation_frames, 1, 1)),
            th.FloatTensor([0.229, 0.224, 0.225]).to(self.device).reshape(
                3, 1, 1).tile((config.n_observation_frames, 1, 1)))
        all_pov_normalization_factors = {
            'MineRLBasaltBuildVillageHouse-v0': (
                th.FloatTensor([109.01, 108.78, 95.13]).to(self.device).reshape(
                    3, 1, 1).tile((config.n_observation_frames, 1, 1)),
                th.FloatTensor([50.83, 56.32, 77.89]).to(self.device).reshape(
                    3, 1, 1).tile((config.n_observation_frames, 1, 1))),
            'MineRLBasaltCreateVillageAnimalPen-v0': (
                th.FloatTensor([107.61, 125.33, 112.16]).to(self.device).reshape(
                    3, 1, 1).tile((config.n_observation_frames, 1, 1)),
                th.FloatTensor([43.69, 50.70, 93.10]).to(self.device).reshape(
                    3, 1, 1).tile((config.n_observation_frames, 1, 1))),
            'MineRLBasaltFindCave-v0': (
                th.FloatTensor([106.44, 127.52, 126.61]).to(self.device).reshape(
                    3, 1, 1).tile((config.n_observation_frames, 1, 1)),
                th.FloatTensor([45.06, 54.25, 97.68]).to(self.device).reshape(
                    3, 1, 1).tile((config.n_observation_frames, 1, 1))),
            'MineRLBasaltMakeWaterfall-v0': (
                th.FloatTensor([109.11, 117.04, 131.58]).to(self.device).reshape(
                    3, 1, 1).tile((config.n_observation_frames, 1, 1)),
                th.FloatTensor([51.78, 60.46, 85.87]).to(self.device).reshape(
                    3, 1, 1).tile((config.n_observation_frames, 1, 1))),
            'MineRLNavigateExtremeDense-v0': (
                th.FloatTensor([70.69, 71.73, 88.11]).to(self.device).reshape(
                    3, 1, 1).tile((config.n_observation_frames, 1, 1)),
                th.FloatTensor([43.07, 49.05, 72.84]).to(self.device).reshape(
                    3, 1, 1).tile((config.n_observation_frames, 1, 1))),
            'MineRLNavigateDense-v0': (
                th.FloatTensor([66.89, 76.77, 104.20]).to(self.device).reshape(
                    3, 1, 1).tile((config.n_observation_frames, 1, 1)),
                th.FloatTensor([54.26, 60.83, 88.11]).to(self.device).reshape(
                    3, 1, 1).tile((config.n_observation_frames, 1, 1))),
        }
        self.pov_normalization = all_pov_normalization_factors[self.env]

    def normalize_state(self, state):
        state = list(state)
        if self.normalize_obs:
            state[0] = (state[0] - self.pov_normalization[0])/self.pov_normalization[1]
            state[0] = state[0] * self.mobilenet_normalization[1] \
                + self.mobilenet_normalization[0]
            state[1] /= self.item_normalization
            if self.env not in \
                    ['MineRLNavigateDense-v0', 'MineRLNavigateExtremeDense-v0']:
                state[1] -= 0.5
        else:
            state[0] /= 255.0
            state[1] /= self.item_normalization
        return tuple(state)

    def states_from_sequence(self, sequence):
        current_states = [state_component[:, :-1, ...] for state_component in sequence]
        next_states = [state_component[:, 1:, ...] for state_component in sequence]
        return current_states, next_states

    def state_to_device(self, state):
        state = [state_component.unsqueeze(0).to(self.device, dtype=th.float)
                 for state_component in state]
        state = self.normalize_state(state)
        # if state includes hidden, add sequence dimension
        if len(state) == 3:
            state = [state_component.unsqueeze(0) for state_component in state]
        return state

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
            states.append(state)
        return tuple(states)

    def expert_batch_to_device(self, batch):
        return self.batch_to_device(batch)

    def batch_to_device(self, batch):
        states, actions, next_states, done, rewards = batch
        states, next_states = self.states_to_device((states, next_states))
        if self.loading_sequences:
            states, next_states = self.states_from_sequence(states)
        actions = actions.unsqueeze(-1).to(self.device)
        done = th.as_tensor(done).unsqueeze(-1).float().to(self.device)
        rewards = rewards.float().unsqueeze(-1).to(self.device)
        return states, actions, next_states, done, rewards

    def batches_to_device(self, expert_batch, replay_batch):
        expert_states, expert_actions, expert_next_states, \
            expert_done, _expert_rewards = expert_batch
        replay_states, replay_actions, replay_next_states, \
            replay_done, replay_rewards = replay_batch

        expert_actions = expert_actions.unsqueeze(-1).to(self.device)
        replay_actions = replay_actions.unsqueeze(-1).to(self.device)

        expert_states, replay_states, expert_next_states, replay_next_states = \
            self.states_to_device((expert_states, replay_states,
                                   expert_next_states, replay_next_states))

        if self.loading_sequences:
            expert_states, expert_next_states = self.states_from_sequence(expert_states)
            replay_states, replay_next_states = self.states_from_sequence(replay_states)

        expert_done = th.as_tensor(expert_done).float().unsqueeze(-1).to(self.device)
        replay_done = th.as_tensor(replay_done).float().unsqueeze(-1).to(self.device)
        replay_rewards = replay_rewards.float().unsqueeze(-1).to(self.device)

        expert_batch = expert_states, expert_actions, expert_next_states, \
            expert_done, _expert_rewards
        replay_batch = replay_states, replay_actions, replay_next_states, \
            replay_done, replay_rewards
        return expert_batch, replay_batch
