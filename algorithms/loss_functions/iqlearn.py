from core.state import cat_states, cat_transitions

import torch as th


class IQLearnLoss:
    def __init__(self, model, config, target_q=None):
        self.model = model
        self.online_q = model
        self.config = config
        self.target_q = target_q
        self.discount_factor = config.method.discount_factor
        self.drq = config.method.drq
        self.loss_type = config.method.loss
        self.online = config.method.online
        self.expert_done_value = config.method.expert_done_value
        if self.online:
            self.policy_done_value = config.method.policy_done_value

    def distance_function(self, x):
        return x - 1/2 * x**2

    def average_across_augmentation(self, tensor):
        aug, no_aug = tensor.chunk(2, dim=0)
        avg = (aug + no_aug) / 2
        return th.cat((avg, avg), dim=0)

    def __call__(self, expert, policy=None, expert_aug=None, policy_aug=None):
        if self.drq:
            expert = cat_transitions((expert, expert_aug))
            if self.online:
                policy = cat_transitions((policy, policy_aug))

        expert_states, expert_actions, _expert_rewards, expert_next_states, \
            expert_done = expert
        if self.online:
            policy_states, policy_actions, _policy_rewards, policy_next_states, \
                policy_done = policy

        if not self.online:
            batch_states, state_lengths = cat_states((expert_states,
                                                      expert_next_states))
            batch_Qs, final_hidden = self.model.get_Q(batch_states)
            if final_hidden.size()[0] != 0:
                final_hidden, _ = final_hidden.chunk(2, dim=0)

            current_Qs_expert, _ = th.split(batch_Qs, state_lengths, dim=0)

            batch_Vs = self.model.get_V(batch_Qs)
            V_expert, V_next_expert = th.split(batch_Vs, state_lengths, dim=0)
        elif self.target_q:
            # get current Q, V with online q
            current_states, current_state_lengths = cat_states((expert_states,
                                                                policy_states))
            current_Qs, final_hidden = self.online_q.get_Q(current_states)
            current_Qs_expert, current_Qs_policy = th.split(
                current_Qs, current_state_lengths, dim=0)

            current_Vs = self.online_q.get_V(current_Qs)
            V_expert, V_policy = th.split(current_Vs, current_state_lengths, dim=0)

            # get next Q, V with online q
            next_states, next_state_lengths = cat_states((expert_next_states,
                                                          policy_next_states))
            with th.no_grad():
                next_Qs, _ = self.target_q.get_Q(next_states)
                next_Vs = self.target_q.get_V(next_Qs)

            V_next_expert, V_next_policy = th.split(next_Vs, next_state_lengths, dim=0)
            batch_Qs = th.cat((current_Qs, next_Qs))

        else:
            # standard online IQ-Learn
            batch_states, state_lengths = cat_states((expert_states, policy_states,
                                                      expert_next_states,
                                                      policy_next_states))
            batch_Qs, final_hidden = self.model.get_Q(batch_states)
            if final_hidden.size()[0] != 0:
                final_hidden, _ = final_hidden.chunk(2, dim=0)

            current_Qs_expert, current_Qs_policy, _, _ = \
                th.split(batch_Qs, state_lengths, dim=0)

            batch_Vs = self.model.get_V(batch_Qs)
            V_expert, V_policy, V_next_expert, V_next_policy = th.split(
                batch_Vs, state_lengths, dim=0)

        Q_s_a_expert = th.gather(current_Qs_expert, dim=-1, index=expert_actions)
        target_Q_expert = (1 - expert_done) * self.discount_factor * V_next_expert \
            + self.expert_done_value * expert_done
        if self.drq:
            target_Q_expert = self.average_across_augmentation(target_Q_expert)

        if self.online:
            Q_s_a_policy = th.gather(current_Qs_policy, dim=-1, index=policy_actions)
            target_Q_policy = (1 - policy_done) * self.discount_factor * V_next_policy \
                + self.policy_done_value * policy_done
            if self.drq:
                target_Q_policy = self.average_across_augmentation(target_Q_policy)

        metrics = {}
        loss = 0

        # keep track of v0
        v0 = V_expert.mean()
        metrics['v0'] = v0

        # keep track of entropy
        with th.no_grad():
            entropy = self.model.batch_entropy(batch_Qs)
        metrics['entropy'] = entropy

        # calculate loss
        loss_expert = -th.mean(self.distance_function(Q_s_a_expert - target_Q_expert))
        # Use an additional regularization term if using target updates
        if self.target_q is not None:
            loss_expert += -th.mean(-1/2*(Q_s_a_policy - target_Q_policy)**2)

        loss += loss_expert
        metrics['softq_loss'] = loss_expert

        if self.loss_type == "v0":
            # calculate 2nd term for our loss
            # (1-γ)E_(ρ0)[V(s0)]
            v0_loss = (1 - self.discount_factor) * v0
            loss += v0_loss
            metrics['v0_loss'] = v0_loss

        elif self.loss_type == "value":
            # alternative 2nd term for our loss (use expert and policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = th.mean(th.cat((V_policy, V_expert), dim=0) -
                                 th.cat((target_Q_policy, target_Q_expert), dim=0))
            loss += value_loss
            metrics['value_loss'] = value_loss

        elif self.loss_type == "value_expert":
            # alternative 2nd term for our loss (use expert and policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = th.mean(V_expert - target_Q_expert)
            loss += value_loss
            metrics['value_loss'] = value_loss

        elif self.loss_type == "value_policy":
            # alternative 2nd term for our loss (use only policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = th.mean(V_policy - target_Q_policy)
            loss += value_loss
            metrics['value_policy_loss'] = value_loss

        metrics["total_loss"] = loss
        for k, v in iter(metrics.items()):
            metrics[k] = v.item()

        return loss, metrics, final_hidden
