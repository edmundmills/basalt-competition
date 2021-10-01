from helpers.environment import ObservationSpace, ActionSpace

import torch as th
import torch.nn.functional as F


class SqilLoss:
    def __init__(self, model, config):
        self.model = model
        self.actions = ActionSpace.actions()
        self.alpha = config.alpha
        self.n_observation_frames = config.n_observation_frames
        self.discount_factor = config.method.discount_factor
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    def __call__(self, expert_batch, replay_batch):
        expert_obs, expert_actions, expert_next_obs, _expert_done = expert_batch
        (replay_obs, replay_actions, replay_next_obs,
         _replay_done, replay_rewards) = replay_batch

        expert_actions = expert_actions.unsqueeze(1)
        replay_actions = replay_actions.unsqueeze(1)

        mask = (expert_actions != -1).squeeze()
        expert_actions = expert_actions[mask]

        expert_batch_size = expert_actions.size()[0]
        replay_batch_size = replay_actions.size()[0]

        batch_actions = th.cat([expert_actions,
                                replay_actions], dim=0).to(self.device)

        expert_states = ObservationSpace.obs_to_state(expert_obs)
        expert_next_states = ObservationSpace.obs_to_state(expert_next_obs)
        replay_states = ObservationSpace.obs_to_state(replay_obs)
        replay_next_states = ObservationSpace.obs_to_state(replay_next_obs)
        expert_states = [state_component[mask] for state_component in expert_states]
        expert_next_states = [state_component[mask]
                              for state_component in expert_next_states]

        batch_states = [th.cat(state_component, dim=0).to(self.device) for state_component
                        in zip(expert_states, replay_states,
                               expert_next_states, replay_next_states)]

        expert_rewards = th.ones(expert_batch_size, 1)
        replay_rewards = replay_rewards.float().unsqueeze(1)
        batch_rewards = th.cat([expert_rewards,
                                replay_rewards], dim=0).to(self.device)

        batch_Qs = self.model.get_Q(batch_states)
        current_Qs, next_Qs = th.chunk(batch_Qs, 2, dim=0)
        predicted_Qs = th.gather(current_Qs, 1, batch_actions)

        V_next = self.model.get_V(next_Qs)
        y = batch_rewards + self.discount_factor * V_next

        loss = F.mse_loss(predicted_Qs, y)
        return loss, {}
