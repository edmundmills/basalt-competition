from core.state import cat_states

import torch as th
import torch.nn.functional as F


class SACQLoss:
    def __init__(self, online_q, target_q, config):
        self.online_q = online_q
        self.target_q = target_q
        self.discount_factor = config.method.discount_factor
        # self.double_q = method_config.double_q

    def __call__(self, batch, batch_no_aug=None):
        states, actions, rewards, next_states, done = batch
        # if self.double_q:
        #     Q1_s_a, Q2_s_a = self.online_q.get_Q_s_a(states, actions)
        #     with th.no_grad():
        #         next_Q1s, next_Q2s = self.target_q.get_Q(next_states)
        #         next_Qs = th.min(next_Q1s, next_Q2s)
        #     next_Vs = self.target_q.get_V(next_Qs)
        #     target_Qs = rewards + (1 - done) * self.discount_factor * next_Vs
        #     loss = F.mse_loss(Q1_s_a, target_Qs) + F.mse_loss(Q2_s_a, target_Qs)
        # else:
        Q_s_a, _ = self.online_q.get_Q_s_a(states, actions)
        with th.no_grad():
            next_Qs, _ = self.target_q.get_Q(next_states)
            next_Vs = self.target_q.get_V(next_Qs)
            target_Qs = rewards + (1 - done) * self.discount_factor * next_Vs
        loss = F.mse_loss(Q_s_a, target_Qs)

        metrics = {'q_loss': loss.detach().item(),
                   'average_target_Q': target_Qs.mean().item()}
        return loss, metrics


class SACQLossDRQ(SACQLoss):
    def __init__(self, online_q, target_q, config):
        super().__init__(self, online_q, target_q, config)

    def __call__(self, aug_batch, batch):
        states, actions, rewards, next_states, done = batch
        states_aug, actions_aug, _rewards_aug, next_states_aug, _done_aug = aug_batch

        Q_s_a, _ = self.online_q.get_Q_s_a(states, actions)
        Q_s_a_aug, _ = self.online_q.get_Q_s_a(states_aug, actions_aug)
        with th.no_grad():
            all_next_states, _lengths = cat_states((next_states, next_states_aug))
            all_next_Qs, _ = self.target_q.get_Q(all_next_states)
        all_next_Vs = self.target_q.get_V(all_next_Qs)
        next_Vs, next_Vs_aug = th.chunk(all_next_Vs, 2, dim=0)
        target_Qs_noaug = rewards + (1 - done) * self.discount_factor * next_Vs
        target_Qs_aug = rewards + (1 - done) * self.discount_factor * next_Vs_aug
        target_Qs = (target_Qs_noaug + target_Qs_aug) / 2
        loss = F.mse_loss(Q_s_a, target_Qs)
        loss += F.mse_loss(Q_s_a_aug, target_Qs)

        metrics = {'q_loss': loss.detach().item(),
                   'average_target_Q': target_Qs.mean().item()}
        return loss, metrics


class SACPolicyLoss:
    def __init__(self, policy, online_q, config, pretraining=False):
        self.online_q = online_q
        self.policy = policy
        self.discount_factor = config.method.discount_factor

    def __call__(self, batch):
        states, _actions, _rewards, _next_states, _done = batch
        actor_Qs, final_hidden = self.policy.get_Q(states)
        entropies = self.policy.entropies(actor_Qs)
        action_probabilities = self.policy.action_probabilities(actor_Qs)

        # if self.double_q:
        #     Q1s, Q2s = self.online_q.get_Q(states)
        #     Qs = th.min(Q1s, Q2s)
        # else:
        with th.no_grad():
            Qs, _ = self.online_q.get_Q(states)

        loss = -th.sum((Qs + self.policy.alpha * entropies)
                       * action_probabilities, dim=1, keepdim=True).mean()

        entropy = th.sum(action_probabilities.detach() * entropies.detach(),
                         dim=1, keepdim=True).mean()
        metrics = {'policy_loss': loss.detach().item(),
                   'entropy': entropy.item()}
        return loss, final_hidden, metrics
