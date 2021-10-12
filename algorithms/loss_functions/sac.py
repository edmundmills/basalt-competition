from utils.environment import ObservationSpace, ActionSpace
from utils.gpu import cat_states

import torch as th
import torch.nn.functional as F


class SACQLoss:
    def __init__(self, online_q, target_q, config, pretraining=False):
        self.online_q = online_q
        self.target_q = target_q
        method_config = config.pretraining if pretraining else config.method
        self.discount_factor = method_config.discount_factor
        # self.double_q = method_config.double_q

    def __call__(self, batch):
        states, actions, next_states, done, rewards = batch
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


class SACQLossDRQ:
    def __init__(self, online_q, target_q, config, pretraining=False):
        self.online_q = online_q
        self.target_q = target_q
        method_config = config.pretraining if pretraining else config.method
        self.discount_factor = method_config.discount_factor
        # self.double_q = method_config.double_q

    def __call__(self, batch, aug_batch):
        states, actions, next_states, done, rewards = batch
        states_aug, actions_aug, next_states_aug, _done_aug, _rewards_aug = batch

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
        method_config = config.pretraining if pretraining else config.method
        self.discount_factor = method_config.discount_factor
        self.entropy_tuning = method_config.entropy_tuning
        # self.double_q = method_config.double_q

    def __call__(self, step, batch):
        states, _actions, _next_states, _done, _rewards = batch
        actor_Qs, final_hidden = self.policy.get_Q(states)
        entropies = self.policy.entropies(actor_Qs)
        action_probabilities = self.policy.action_probabilities(actor_Qs)

        # if self.double_q:
        #     Q1s, Q2s = self.online_q.get_Q(states)
        #     Qs = th.min(Q1s, Q2s)
        # else:
        Qs, _ = self.online_q.get_Q(states)

        loss = -th.sum((Qs + self.policy.alpha * entropies)
                       * action_probabilities, dim=1, keepdim=True).mean()
        if self.entropy_tuning:
            alpha_loss = th.sum((-self.log_alpha *
                                 (self.target_entropy - entropies.detach())) *
                                action_probabilities.detach(), dim=1, keepdim=True).mean()
        else:
            alpha_loss = th.tensor([0])
        entropy = th.sum(action_probabilities.detach() * entropies.detach(),
                         dim=1, keepdim=True).mean()
        metrics = {'policy_loss': loss.detach().item(),
                   'alpha_loss': alpha_loss.detach().item(),
                   'entropy': entropy.item()}
        return loss, alpha_loss, final_hidden, metrics


class CuriousIQPolicyLoss:
    def __init__(self, policy, online_q, iqlearn_q, config):
        self.online_q = online_q
        self.iqlearn_q = iqlearn_q
        self.policy = policy
        self.discount_factor = config.method.discount_factor
        self.initial_curiosity_fraction = config.method.initial_curiosity_fraction
        self.curiosity_only_steps = config.method.curiosity_only_steps
        self.curiosity_fade_out_steps = config.method.curiosity_fade_out_steps

    def __call__(self, step, batch):
        steps_in_fade = step - self.curiosity_only_steps
        if steps_in_fade == self.curiosity_fade_out_steps:
            print('Updating actor with iqlearn only')
        elif steps_in_fade == 0:
            print('Updating actor with curiosity and iqlearn')
        if steps_in_fade >= 0 and steps_in_fade <= self.curiosity_fade_out_steps \
                and self.curiosity_fade_out_steps != 0:
            curiosity_fraction = (self.curiosity_fade_out_steps - steps_in_fade) \
                / self.curiosity_fade_out_steps * self.initial_curiosity_fraction
        else:
            curiosity_fraction = None

        states, _actions, _next_states, _done, _rewards = batch
        actor_Qs, _ = self.policy.get_Q(states)
        entropies = self.policy.entropies(actor_Qs)
        action_probabilities = self.policy.action_probabilities(actor_Qs)

        metrics = {}

        if steps_in_fade < self.curiosity_fade_out_steps:
            with th.no_grad():
                curiosity_Qs, _ = self.online_q.get_Q(states)
            curiosity_loss = -th.sum((curiosity_Qs + self.policy.alpha * entropies)
                                     * action_probabilities, dim=1, keepdim=True).mean()
            curiosity_fraction = curiosity_fraction or self.initial_curiosity_fraction
        else:
            curiosity_loss = 0
            curiosity_fraction = 0

        with th.no_grad():
            iqlearn_Qs, _ = self.iqlearn_q.get_Q(states)
        policy_loss = -th.sum((iqlearn_Qs + self.policy.alpha * entropies)
                              * action_probabilities, dim=1, keepdim=True).mean()

        loss = curiosity_fraction * curiosity_loss \
            + (1 - curiosity_fraction) * policy_loss

        alpha_loss = th.sum((-self.log_alpha * (self.target_entropy - entropies.detach()))
                            * action_probabilities.detach(), dim=1, keepdim=True).mean()
        entropy = th.sum(action_probabilities.detach() * entropies.detach(),
                         dim=1, keepdim=True).mean()

        metrics['alpha_loss'] = alpha_loss.detach().item()
        metrics['policy_loss'] = loss.detach().item()
        metrics['entropy'] = entropy.detach().item()
        metrics['curiosity_fraction'] = curiosity_fraction
        return loss, alpha_loss, metrics
