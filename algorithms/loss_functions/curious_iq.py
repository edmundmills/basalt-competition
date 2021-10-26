from algorithms.loss_functions.iqlearn import IQLearnLoss
from core.state import cat_states

import torch as th
import torch.nn.functional as F


class CuriousIQLoss(IQLearnLoss):
    def __init__(self, agent, config):
        super().__init__(agent, config)


class CuriousIQPolicyLoss:
    def __init__(self, policy, online_q, iqlearn_q, config):
        self.online_q = online_q
        self.iqlearn_q = iqlearn_q
        self.policy = policy
        self.discount_factor = config.method.discount_factor
        self.initial_curiosity_fraction = config.method.initial_curiosity_fraction
        self.curiosity_only_steps = config.method.curiosity_only_steps
        self.curiosity_fade_out_steps = config.method.curiosity_fade_out_steps

    def __call__(self, batch):
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
