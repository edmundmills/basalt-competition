from helpers.environment import ObservationSpace, ActionSpace
from helpers.gpu import cat_states

import torch as th
import torch.nn.functional as F


class SACQLoss:
    def __init__(self, online_q, target_q, config, pretraining=False):
        self.online_q = online_q
        self.target_q = target_q
        method_config = config.pretraining if pretraining else config.method
        self.discount_factor = method_config.discount_factor
        self.double_q = method_config.double_q

    def __call__(self, batch):
        states, actions, next_states, done, rewards = batch
        if self.double_q:
            Q1_s_a, Q2_s_a = self.online_q.get_Q_s_a(states, actions)
            with th.no_grad():
                next_Q1s, next_Q2s = self.target_q.get_Q(next_states)
                next_Qs = th.min(next_Q1s, next_Q2s)
            next_Vs = self.target_q.get_V(next_Qs)
            target_Qs = rewards + (1 - done) * self.discount_factor * next_Vs
            loss = F.mse_loss(Q1_s_a, target_Qs) + F.mse_loss(Q2_s_a, target_Qs)
        else:
            Q_s_a = self.online_q.get_Q_s_a(states, actions)
            with th.no_grad():
                next_Qs = self.target_q.get_Q(next_states)
            next_Vs = self.target_q.get_V(next_Qs)
            target_Qs = rewards + (1 - done) * self.discount_factor * next_Vs
            loss = F.mse_loss(Q_s_a, target_Qs)

        metrics = {'q_loss': loss.detach().item(),
                   'average_target_Q': target_Qs.mean().item()}
        return loss, metrics


class SACQLossDRQ:
    def __init__(self, online_q, target_q, config, pretraining=False):
        self.online_q = online_q
        self.target_q = target_q
        method_config = config.pretraining if pretraining else config.method
        self.discount_factor = method_config.discount_factor
        self.double_q = method_config.double_q

    def __call__(self, batch, aug_batch):
        states, actions, next_states, done, rewards = batch
        states_aug, actions_aug, next_states_aug, _done_aug, _rewards_aug = batch

        Q_s_a = self.online_q.get_Q_s_a(states, actions)
        Q_s_a_aug = self.online_q.get_Q_s_a(states_aug, actions_aug)
        with th.no_grad():
            all_next_states, _lengths = cat_states((next_states, next_states_aug))
            all_next_Qs = self.target_q.get_Q(all_next_states)
        all_next_Vs = self.target_q.get_V(all_next_Qs)
        next_Vs, next_Vs_aug = th.chunk(all_next_Vs, 2, dim=0)
        target_Qs_noaug = rewards + (1 - done) * self.discount_factor * next_Vs
        target_Qs_aug = rewards + (1 - done) * self.discount_factor * next_Vs_aug
        target_Qs = (target_Qs_noaug + target_Qs_aug) / 2
        loss = F.mse_loss(Q_s_a, target_Qs)
        loss += F.mse_loss(Q_s_a_aug, target_Qs)

        metrics = {'q_loss': loss.detach().item(),
                   'average_target_Q': target_Qs.mean().item(),
                   'target_Q_noaug': target_Qs_noaug.mean().item(),
                   'target_Q_aug': target_Qs_aug.mean().item()}
        return loss, metrics


class SACPolicyLoss:
    def __init__(self, policy, online_q, config, pretraining=False):
        self.online_q = online_q
        self.policy = policy
        method_config = config.pretraining if pretraining else config.method
        self.discount_factor = method_config.discount_factor
        self.double_q = method_config.double_q

    def __call__(self, batch):
        states, _actions, _next_states, _done, _rewards = batch
        actor_Qs = self.policy.get_Q(states)
        entropies = self.policy.entropy(actor_Qs)
        action_probabilities = self.policy.action_probabilities(actor_Qs)

        if self.double_q:
            Q1s, Q2s = self.online_q.get_Q(states)
            Qs = th.min(Q1s, Q2s)
        else:
            Qs = self.online_q.get_Q(states)
        # this is elementwise multiplication to get expectation of Q for following policy
        expected_Q_policy = th.sum(Qs * action_probabilities, dim=1, keepdim=True)
        # entropy has negative sign: -logpi.
        # whole term is negative since we want to maximize Q and entropy
        loss = -th.mean(expected_Q_policy + self.policy.alpha * entropies)
        metrics = {'policy_loss': loss.detach().item(),
                   'average_online_Q': Qs.detach().mean().item(),
                   'policy_Q': expected_Q_policy.detach().mean().item(),
                   'entropy': entropies.detach().mean().item()}
        return loss, metrics


class CuriousIQPolicyLoss:
    def __init__(self, policy, online_q, iqlearn_q, config):
        self.online_q = online_q
        self.iqlearn_q = iqlearn_q
        self.policy = policy
        self.discount_factor = config.method.discount_factor
        self.curiosity_fraction = 0.5

    def __call__(self, batch):
        states, _actions, _next_states, _done, _rewards = batch
        actor_Qs = self.policy.get_Q(states)
        entropies = self.policy.entropy(actor_Qs)
        action_probabilities = self.policy.action_probabilities(actor_Qs)

        curiosity_Qs = self.online_q.get_Q(states)
        iqlearn_Qs = self.iqlearn_q.get_Q(states)
        Qs = curiosity_Qs * self.curiosity_fraction + \
            iqlearn_Qs * (1 - self.curiosity_fraction)
        # this is elementwise multiplication to get expectation of Q for following policy
        Q_policy = th.sum(Qs * action_probabilities, dim=1, keepdim=True)
        # entropy has negative sign: -logpi.
        # whole term is negative since we want to maximize Q and entropy
        loss = -th.mean(Q_policy + self.policy.alpha * entropies)
        metrics = {'policy_loss': loss.detach().item(),
                   'curiosity_Q': curiosity_Qs.detach().mean().item(),
                   'iqlearn_Q': iqlearn_Qs.detach().mean().item(),
                   'average_Q': Qs.detach().mean().item(),
                   'policy_Q': Q_policy.detach().mean().item(),
                   'entropy': entropies.detach().mean().item()}
        return loss, metrics
