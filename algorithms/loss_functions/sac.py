from helpers.environment import ObservationSpace, ActionSpace

import torch as th
import torch.nn.functional as F


class SACQLoss:
    def __init__(self, online_q, target_q, config):
        self.online_q = online_q
        self.target_q = target_q
        self.discount_factor = config.discount_factor
        self.double_q = config.double_q

    def __call__(self, states, actions, next_states, done, rewards):
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


class SACPolicyLoss:
    def __init__(self, policy, online_q, config):
        self.online_q = online_q
        self.policy = policy
        self.discount_factor = config['discount_factor']
        self.double_q = config.double_q

    def __call__(self, states, _actions, _next_states, _done, _rewards):
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
