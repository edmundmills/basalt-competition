from helpers.environment import ObservationSpace, ActionSpace
from helpers.gpu import cat_states

import torch as th
import torch.nn.functional as F


class IQLearnLoss:
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.actions = ActionSpace.actions()
        self.alpha = config.alpha
        self.discount_factor = config.method.discount_factor
        self.drq = config.method.drq
        self.n_observation_frames = config.n_observation_frames

    def __call__(self, expert_batch, replay_batch):
        expert_states, expert_actions, expert_next_states, \
            expert_done, _expert_rewards = expert_batch
        replay_states, replay_actions, replay_next_states, \
            replay_done, _replay_rewards = replay_batch

        batch_states = expert_states, replay_states, \
            expert_next_states, replay_next_states
        batch_states, state_lengths = cat_states(batch_states)
        batch_Qs = self.model.get_Q(batch_states)
        Q_expert, _, _, _ = th.split(batch_Qs, state_lengths, dim=0)

        predicted_Q_expert = th.gather(Q_expert, 1, expert_actions)

        batch_Vs = self.model.get_V(batch_Qs)
        V_expert, V_replay, V_next_expert, V_next_replay = th.split(
            batch_Vs, state_lengths, dim=0)

        metrics = {}

        # keep track of v0
        v0 = V_expert.mean()
        v0_aug = V_expert_aug.mean()
        metrics['v0'] = v0
        metrics['v0_aug'] = v0_aug

        def distance_function(x):
            return x - 1/4 * x**2

        loss_expert = -th.mean(distance_function(
            predicted_Q_expert - (1 - expert_done) * self.discount_factor * V_next_expert)
        )
        loss = loss_expert
        metrics['softq_loss'] = loss_expert

        if self.config.method.loss == "v0":
            # calculate 2nd term for our loss
            # (1-γ)E_(ρ0)[V(s0)]
            v0_loss = (1 - self.discount_factor) * v0
            v0_loss += (1 - self.discount_factor) * v0_aug
            loss += v0_loss
            metrics['v0_loss'] = v0_loss

        elif self.config.method.loss == "value":
            # alternative 2nd term for our loss (use expert and policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            done = th.cat((replay_done, expert_done), dim=0)
            value_loss = th.mean(th.cat((V_replay, V_expert), dim=0) -
                                 (1 - done) * self.discount_factor *
                                 th.cat((V_next_replay, V_next_expert), dim=0))
            loss += value_loss
            metrics['value_loss'] = value_loss

        elif self.config.method.loss == "value_expert":
            # alternative 2nd term for our loss (use expert and policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = th.mean(V_expert - (1 - expert_done) *
                                 self.discount_factor * V_next_expert)
            loss += value_loss
            metrics['value_loss'] = value_loss

        elif self.config.method.loss == "value_policy":
            # alternative 2nd term for our loss (use only policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = th.mean(V_replay - (1 - replay_done) *
                                 self.discount_factor * V_next_replay)
            loss += value_loss
            metrics['value_policy_loss'] = value_loss

        metrics.update({
            "total_loss": loss,
        })
        for k, v in iter(metrics.items()):
            metrics[k] = v.item()
        return loss, metrics


class IQLearnLossDRQ(IQLearnLoss):
    def __init__(self, model, config):
        super().__init__(model, config)

    def __call__(self, expert_batch, replay_batch, aug_exp_batch, aug_rep_batch):
        expert_states, expert_actions, expert_next_states, \
            expert_done, _expert_rewards = expert_batch
        replay_states, replay_actions, replay_next_states, \
            replay_done, _replay_rewards = replay_batch
        expert_states_aug, expert_actions_aug, expert_next_states_aug, \
            _expert_done_aug, _expert_rewards = expert_batch
        replay_states_aug, replay_actions_aug, replay_next_states_aug, \
            _replay_done_aug, _replay_rewards = replay_batch

        batch_states = expert_states, replay_states, \
            expert_next_states, replay_next_states, \
            expert_states_aug, replay_states_aug, \
            expert_next_states_aug, replay_next_states_aug

        batch_states, state_lengths = cat_states(batch_states)
        batch_Qs = self.model.get_Q(batch_states)
        Q_expert, _, _, _, Q_expert_aug, _, _, _ = th.split(batch_Qs,
                                                            state_lengths, dim=0)

        predicted_Q_expert = th.gather(Q_expert, 1, expert_actions)
        predicted_Q_expert_aug = th.gather(Q_expert_aug, 1, expert_actions_aug)

        batch_Vs = self.model.get_V(batch_Qs)
        V_expert, V_replay, V_next_expert, V_next_replay, \
            V_expert_aug, V_replay_aug, V_next_expert_aug, V_next_replay_aug = th.split(
                batch_Vs, state_lengths, dim=0)

        metrics = {}

        # keep track of v0
        v0 = V_expert.mean()
        metrics['v0'] = v0

        def distance_function(x):
            return x - 1/4 * x**2

        target_Q_exp = (1 - expert_done) * self.discount_factor * V_next_expert
        target_Q_exp_aug = (1 - expert_done) * self.discount_factor * V_next_expert_aug
        target_Q_exp = (target_Q_exp + target_Q_exp_aug) / 2

        target_Q_rep = (1 - replay_done) * self.discount_factor * V_next_replay
        target_Q_rep_aug = (1 - replay_done) * self.discount_factor * V_next_replay_aug
        target_Q_rep = (target_Q_rep + target_Q_rep_aug) / 2

        loss_expert = -th.mean(distance_function(predicted_Q_expert - target_Q_exp))
        loss_expert += -th.mean(distance_function(predicted_Q_expert_aug - target_Q_exp))
        loss = loss_expert
        metrics['softq_loss'] = loss_expert

        if self.config.method.loss == "v0":
            # calculate 2nd term for our loss
            # (1-γ)E_(ρ0)[V(s0)]
            v0_loss = (1 - self.discount_factor) * v0
            loss += v0_loss
            metrics['v0_loss'] = v0_loss

        elif self.config.method.loss == "value":
            # alternative 2nd term for our loss (use expert and policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = th.mean(th.cat((V_replay, V_expert), dim=0) -
                                 th.cat((target_Q_rep, target_Q_exp), dim=0))
            value_loss += th.mean(th.cat((V_replay_aug, V_expert_aug), dim=0) -
                                  th.cat((target_Q_rep, target_Q_exp), dim=0))
            loss += value_loss
            metrics['value_loss'] = value_loss

        elif self.config.method.loss == "value_expert":
            # alternative 2nd term for our loss (use expert and policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = th.mean(V_expert - target_Q_exp)
            value_loss += th.mean(V_expert_aug - target_Q_exp)
            loss += value_loss
            metrics['value_loss'] = value_loss

        elif self.config.method.loss == "value_policy":
            # alternative 2nd term for our loss (use only policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = th.mean(V_replay - target_Q_rep)
            value_loss += th.mean(V_replay_aug - target_Q_rep)
            loss += value_loss
            metrics['value_policy_loss'] = value_loss

        metrics.update({
            "total_loss": loss,
        })
        for k, v in iter(metrics.items()):
            metrics[k] = v.item()
        return loss, metrics


class IQLearnLossSAC(IQLearnLoss):
    def __init__(self, model, config, target_q):
        super().__init__(model, config)
        self.target_q = target_q
        self.online_q = self.model

    def __call__(self, expert_states, expert_actions, expert_next_states, expert_done,
                 replay_states, _replay_actions, replay_next_states, replay_done):
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
