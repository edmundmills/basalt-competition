from helpers.environment import ObservationSpace, ActionSpace

import torch as th
import torch.nn.functional as F


class SACQLoss:
    def __init__(self, critic, run):
        self.critic = model
        self.discount_factor = run.config['discount_factor']

    def __call__(self, all_states, actions):
        all_Qs = self.critic.get_Q(all_states)
        Qs, next_Qs = th.chunk(all_Qs, 2, dim=0)
        next_Vs = self.critic.get_V(next_Qs)
        # use Qs only for taken actions
        Q_s_a = th.gather(Qs, dim=1, index=actions.unsqueeze(1))
        target_Qs = rewards + self.discount_factor * next_Vs
        loss = F.mse_loss(Q_s_a, target_Qs)
        return loss


class SACPolicyLoss:
    def __init__(self, actor, critic, run):
        self.actor = actor
        self.critic = critic
        self.discount_factor = run.config['discount_factor']

    def __call__(self, states):
        entropies = self.actor.entropies(Qs)
        entropy_s_a = th.gather(entropies, dim=1, index=actions.unsqueeze(1))
        loss = -th.mean(Q_s_a - self.actor.alpha * entropy_s_a)
        return loss
