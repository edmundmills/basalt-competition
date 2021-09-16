from helpers.environment import ObservationSpace, ActionSpace

import torch as th
import torch.nn.functional as F


class SACQLoss:
    def __init__(self, online_q1, target_q2, run):
        self.online_q = online_q
        self.target_q = target_q
        self.discount_factor = run.config['discount_factor']

    def __call__(self, states, actions, next_states, _done, rewards):
        Q1_s_a, Q2_s_a = self.online_q.get_Q_s_a(states, actions)
        with th.no_grad():
            next_Q1s, next_Q2s = self.target_q.get_Q(next_states)
            next_Qs = th.min(next_Q1s, next_Q2s)
            next_Vs = self.target_q.get_V(next_Qs)
            target_Qs = rewards + self.discount_factor * next_Vs
        loss = F.mse_loss(Q1_s_a, target_Qs) + F.mse_loss(Q2_s_a, target_Qs)
        return loss, target_Qs.mean().item()


class SACPolicyLoss:
    def __init__(self, policy, online_q, run):
        self.online_q = online_q
        self.policy = policy
        self.discount_factor = run.config['discount_factor']

    def __call__(self, states, _actions, _next_states, _done, _rewards):
        Q1s, Q2s = self.online_q.get_Q(states)
        Qs = th.min(Q1s, Q2s)
        # this is elementwise multiplication to get expectation of Q for following policy
        expected_Q_policy = Qs * self.policies.action_probabilities(states)
        entropies = self.policy.entropies(Qs)
        entropy_s_a = th.gather(entropies, dim=1, index=actions.unsqueeze(1))
        loss = -th.mean(expected_Q_policy - self.policy.alpha * entropy_s_a)
        return loss, Qs.detach().mean().item()
