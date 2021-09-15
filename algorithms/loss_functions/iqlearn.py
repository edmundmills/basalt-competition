from helpers.environment import ObservationSpace, ActionSpace

import torch as th
import torch.nn.functional as F


class IQLearn:
    def __init__(self, model, run):
        self.model = model
        self.actions = ActionSpace.actions()
        self.alpha = run.config['alpha']
        self.discount_factor = run.config['discount_factor']
        self.n_observation_frames = run.config['n_observation_frames']
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    def __call__(self, expert_batch, replay_batch):
        expert_obs, expert_actions, expert_next_obs, _expert_done = expert_batch
        (replay_obs, replay_actions, replay_next_obs,
         _replay_done, replay_rewards) = replay_batch

        expert_actions = ActionSpace.dataset_action_batch_to_actions(expert_actions)
        expert_actions = th.from_numpy(expert_actions).unsqueeze(1)
        replay_actions = replay_actions.unsqueeze(1)

        mask = (expert_actions != -1).squeeze()
        expert_actions = expert_actions[mask].to(self.device)

        expert_batch_size = expert_actions.size()[0]
        replay_batch_size = replay_actions.size()[0]

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

        batch_lengths = [expert_batch_size, replay_batch_size,
                         expert_batch_size, replay_batch_size]

        batch_Qs = self.model.get_Q(batch_states)

        Q_expert, _, _, _ = th.split(batch_Qs, batch_lengths, dim=0)
        predicted_Q_expert = th.gather(Q_expert, 1, expert_actions)

        batch_Vs = self.model.get_V(batch_Qs)
        V_expert, V_replay, V_next_expert, V_next_replay = th.split(
            batch_Vs, batch_lengths, dim=0)

        def distance_function(x):
            return x - 1/4 * x**2

        loss = -(th.mean(distance_function(predicted_Q_expert -
                                           self.discount_factor * V_next_expert)) -
                 th.mean(th.cat((V_replay, V_expert), dim=0) -
                         self.discount_factor * th.cat((V_next_replay,
                                                        V_next_expert), dim=0)))

        # # Add an additional term to the loss for the reward of the throw snoball actions
        # replay_rewards = replay_rewards.unsqueeze(1).float().to(self.device)
        # rewards_mask = replay_rewards != 0.
        # replay_rewards = replay_rewards[rewards_mask]
        # if replay_rewards.size()[0] > 0:
        #     predicted_Q_replay = th.gather(Q_replay, 1, replay_actions)
        #     predicted_r = (predicted_Q_replay[rewards_mask]
        #                    - self.discount_factor * V_next_replay[rewards_mask])
        #     objective += F.mse_loss(predicted_r, replay_rewards)
        return loss
