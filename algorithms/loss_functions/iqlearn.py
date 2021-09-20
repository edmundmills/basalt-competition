from helpers.environment import ObservationSpace, ActionSpace
from helpers.gpu import states_to_device, cat_states

import torch as th
import torch.nn.functional as F


class IQLearnLoss:
    def __init__(self, model, config):
        self.model = model
        self.actions = ActionSpace.actions()
        self.alpha = config['alpha']
        self.discount_factor = config['discount_factor']
        self.n_observation_frames = config['n_observation_frames']

    def batches_to_device(self, expert_batch, replay_batch):
        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        expert_obs, expert_actions, expert_next_obs, _expert_done = expert_batch
        (replay_obs, replay_actions, replay_next_obs,
         _replay_done, _replay_rewards) = replay_batch

        expert_actions = ActionSpace.dataset_action_batch_to_actions(expert_actions)
        expert_actions = th.from_numpy(expert_actions).unsqueeze(1)
        replay_actions = replay_actions.unsqueeze(1)

        mask = (expert_actions != -1).squeeze()
        expert_actions = expert_actions[mask].to(device)

        expert_states = ObservationSpace.obs_to_state(expert_obs)
        expert_next_states = ObservationSpace.obs_to_state(expert_next_obs)
        replay_states = ObservationSpace.obs_to_state(replay_obs)
        replay_next_states = ObservationSpace.obs_to_state(replay_next_obs)
        expert_states = [state_component[mask] for state_component in expert_states]
        expert_next_states = [state_component[mask]
                              for state_component in expert_next_states]

        batch_states = states_to_device((expert_states, replay_states,
                                        expert_next_states, replay_next_states),
                                        device)
        return batch_states, expert_actions, replay_actions

    def __call__(self, expert_batch, replay_batch):
        batch_states, expert_actions, _replay_actions = self.batches_to_device(
            expert_batch, replay_batch)

        batch_states, state_lengths = cat_states(batch_states)

        batch_Qs = self.model.get_Q(batch_states)

        Q_expert, _, _, _ = th.split(batch_Qs, state_lengths, dim=0)
        predicted_Q_expert = th.gather(Q_expert, 1, expert_actions)

        batch_Vs = self.model.get_V(batch_Qs)
        V_expert, V_replay, V_next_expert, V_next_replay = th.split(
            batch_Vs, state_lengths, dim=0)

        def distance_function(x):
            return x - 1/4 * x**2

        loss = -(th.mean(distance_function(predicted_Q_expert -
                                           self.discount_factor * V_next_expert)) -
                 th.mean(th.cat((V_replay, V_expert), dim=0) -
                         self.discount_factor * th.cat((V_next_replay,
                                                        V_next_expert), dim=0)))
        return loss


class IQLearnLossSAC(IQLearnLoss):
    def __init__(self, model, config, target_q):
        super().__init__(model, config)
        self.target_q = target_q
        self.online_q = self.model

    def __call__(self, expert_states, expert_actions, expert_next_states,
                 replay_states, _replay_actions, replay_next_states):
        current_states, current_state_lengths = cat_states((expert_states, replay_states))
        current_Qs = self.online_q.get_Q(current_states)
        current_Vs = self.online_q.get_V(current_Qs)
        current_Qs_expert, _ = th.split(current_Qs, current_state_lengths, dim=0)
        Q_s_a_expert = th.gather(current_Qs_expert, dim=1, index=expert_actions)

        next_states, next_state_lengths = cat_states((expert_next_states,
                                                      replay_next_states))
        with th.no_grad():
            next_Qs = self.target_q.get_Q(next_states)
            next_Vs = self.target_q.get_V(next_Qs)
            V_next_expert, _ = th.split(next_Vs, next_state_lengths, dim=0)

        def distance_function(x):
            return x - 1/4 * x**2

        loss = -(th.mean(distance_function(Q_s_a_expert -
                                           self.discount_factor * V_next_expert)) -
                 th.mean(current_Vs - self.discount_factor * next_Vs))

        metrics = {'q_loss': loss.detach().item(),
                   'average_target_Q': next_Qs.detach().mean().item()}
        return loss, metrics
