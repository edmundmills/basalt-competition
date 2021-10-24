from core.state import cat_states

import torch as th


class IQLearnLoss:
    def __init__(self, model, config, target_q=None):
        self.model = model
        self.config = config
        self.target_q = target_q
        self.discount_factor = config.method.discount_factor
        self.drq = config.method.drq

    def __call__(self, expert_batch, replay_batch):
        expert_states, expert_actions, expert_next_states, \
            expert_done, _expert_rewards = expert_batch
        replay_states, replay_actions, replay_next_states, \
            replay_done, _replay_rewards = replay_batch

        if self.target_q is None:
            batch_states = expert_states, replay_states, \
                expert_next_states, replay_next_states
            batch_states, state_lengths = cat_states(batch_states)
            batch_Qs, final_hidden = self.model.get_Q(batch_states)
            if final_hidden is not None:
                final_hidden, _ = final_hidden.chunk(2, dim=0)
            Q_expert, _, _, _ = th.split(batch_Qs, state_lengths, dim=0)

            predicted_Q_expert = th.gather(Q_expert, -1, expert_actions)

            batch_Vs = self.model.get_V(batch_Qs)
            V_expert, V_replay, V_next_expert, V_next_replay = th.split(
                batch_Vs, state_lengths, dim=0)
            with th.no_grad():
                entropies = self.model.entropies(batch_Qs)
                action_probabilities = self.model.action_probabilities(batch_Qs)
        else:
            current_states, current_state_lengths = cat_states((expert_states,
                                                                replay_states))
            current_Qs, final_hidden = self.model.get_Q(current_states)
            current_Vs = self.model.get_V(current_Qs)
            current_Qs_expert, current_Qs_replay = th.split(
                current_Qs, current_state_lengths, dim=0)
            predicted_Q_expert = th.gather(current_Qs_expert, dim=-1,
                                           index=expert_actions)
            predicted_Q_replay = th.gather(current_Qs_replay, dim=-1,
                                           index=replay_actions)

            next_states, next_state_lengths = cat_states((expert_next_states,
                                                          replay_next_states))
            with th.no_grad():
                next_Qs, _ = self.target_q.get_Q(next_states)
                next_Vs = self.target_q.get_V(next_Qs)

            V_expert, V_replay = th.split(current_Vs, current_state_lengths, dim=0)
            V_next_expert, V_next_replay = th.split(next_Vs, next_state_lengths, dim=0)
            with th.no_grad():
                entropies = self.model.entropies(current_Qs)
                action_probabilities = self.model.action_probabilities(current_Qs)

        metrics = {}

        target_Q_exp = (1 - expert_done) * self.discount_factor * V_next_expert
        target_Q_rep = (1 - replay_done) * self.discount_factor * V_next_replay

        # keep track of v0
        v0 = V_expert.mean()
        metrics['v0'] = v0

        def distance_function(x):
            return x - 1/2 * x**2

        loss_expert = -th.mean(distance_function(predicted_Q_expert - target_Q_exp))
        if self.target_q is not None:
            loss_expert += -th.mean(-1/2*(predicted_Q_replay - target_Q_rep)**2)

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
            value_loss = th.mean(th.cat((V_replay, V_expert), dim=0) -
                                 th.cat((target_Q_rep, target_Q_exp), dim=0))
            loss += value_loss
            metrics['value_loss'] = value_loss

        elif self.config.method.loss == "value_expert":
            # alternative 2nd term for our loss (use expert and policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = th.mean(V_expert - target_Q_exp)
            loss += value_loss
            metrics['value_loss'] = value_loss

        elif self.config.method.loss == "value_policy":
            # alternative 2nd term for our loss (use only policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = th.mean(V_replay - target_Q_rep)
            loss += value_loss
            metrics['value_policy_loss'] = value_loss

        entropy = th.sum(action_probabilities.detach() * entropies.detach(),
                         dim=1, keepdim=True).mean()

        metrics.update({
            "total_loss": loss,
            'entropy': entropy,
        })
        for k, v in iter(metrics.items()):
            metrics[k] = v.item()
        return loss, metrics, final_hidden


class IQLearnLossDRQ(IQLearnLoss):
    def __init__(self, model, config, target_q=None):
        super().__init__(model, config, target_q=target_q)

    def __call__(self, expert_batch, replay_batch, aug_exp_batch, aug_rep_batch):
        expert_states, expert_actions, expert_next_states, \
            expert_done, _expert_rewards = expert_batch
        replay_states, replay_actions, replay_next_states, \
            replay_done, _replay_rewards = replay_batch
        expert_states_aug, expert_actions_aug, expert_next_states_aug, \
            _expert_done_aug, _expert_rewards = expert_batch
        replay_states_aug, replay_actions_aug, replay_next_states_aug, \
            _replay_done_aug, _replay_rewards = replay_batch

        if self.target_q is None:
            batch_states = expert_states, replay_states, \
                expert_next_states, replay_next_states, \
                expert_states_aug, replay_states_aug, \
                expert_next_states_aug, replay_next_states_aug

            batch_states, state_lengths = cat_states(batch_states)
            batch_Qs, final_hidden = self.model.get_Q(batch_states)
            if final_hidden is not None:
                final_hidden, _, _, _ = final_hidden.chunk(4, dim=0)
            Q_expert, _, _, _, Q_expert_aug, _, _, _ = th.split(batch_Qs,
                                                                state_lengths, dim=0)

            predicted_Q_expert = th.gather(Q_expert, -1, expert_actions)
            predicted_Q_expert_aug = th.gather(Q_expert_aug, -1, expert_actions_aug)

            batch_Vs = self.model.get_V(batch_Qs)

            V_expert, V_replay, V_next_expert, V_next_replay, V_expert_aug, \
                V_replay_aug, V_next_expert_aug, V_next_replay_aug = th.split(
                    batch_Vs, state_lengths, dim=0)
            with th.no_grad():
                entropies = self.model.entropies(batch_Qs)
                action_probabilities = self.model.action_probabilities(batch_Qs)
        else:
            current_states, current_state_lengths = cat_states((expert_states,
                                                                replay_states,
                                                                expert_states_aug,
                                                                replay_states_aug))
            current_Qs, final_hidden = self.model.get_Q(current_states)
            if final_hidden is not None:
                final_hidden, _ = final_hidden.chunk(2, dim=0)
            current_Vs = self.model.get_V(current_Qs)
            Q_expert, Q_replay, Q_expert_aug, Q_replay_aug = th.split(
                current_Qs, current_state_lengths, dim=0)
            predicted_Q_expert = th.gather(Q_expert, -1, expert_actions)
            predicted_Q_expert_aug = th.gather(Q_expert_aug, -1, expert_actions_aug)
            predicted_Q_replay = th.gather(Q_replay, -1, replay_actions)
            predicted_Q_replay_aug = th.gather(Q_replay_aug, -1, replay_actions_aug)

            next_states, next_state_lengths = cat_states((expert_next_states,
                                                          replay_next_states,
                                                          expert_next_states_aug,
                                                          replay_next_states_aug))
            with th.no_grad():
                next_Qs, _ = self.target_q.get_Q(next_states)
                next_Vs = self.target_q.get_V(next_Qs)

            V_expert, V_replay, V_expert_aug, V_replay_aug = th.split(
                current_Vs, current_state_lengths, dim=0)
            V_next_expert, V_next_replay, V_next_expert_aug, V_next_replay_aug = th.split(
                next_Vs, next_state_lengths, dim=0)
            with th.no_grad():
                entropies = self.model.entropies(current_Qs)
                action_probabilities = self.model.action_probabilities(current_Qs)

        metrics = {}

        # keep track of v0
        v0 = V_expert.mean()
        metrics['v0'] = v0
        v0_aug = V_expert_aug.mean()
        metrics['v0_aug'] = v0_aug

        def distance_function(x):
            return x - 1/2 * x**2

        target_Q_exp = (1 - expert_done) * self.discount_factor * V_next_expert
        target_Q_exp_aug = (1 - expert_done) * self.discount_factor * V_next_expert_aug
        target_Q_exp = (target_Q_exp + target_Q_exp_aug) / 2

        target_Q_rep = (1 - replay_done) * self.discount_factor * V_next_replay
        target_Q_rep_aug = (1 - replay_done) * self.discount_factor * V_next_replay_aug
        target_Q_rep = (target_Q_rep + target_Q_rep_aug) / 2

        loss_expert = -th.mean(distance_function(predicted_Q_expert - target_Q_exp))
        loss_expert += -th.mean(distance_function(predicted_Q_expert_aug - target_Q_exp))
        if self.target_q is not None:
            loss_expert += -th.mean(-1/2*(predicted_Q_replay - target_Q_rep)**2)
            loss_expert += -th.mean(-1/2*(predicted_Q_replay_aug - target_Q_rep)**2)

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

        entropy = th.sum(action_probabilities.detach() * entropies.detach(),
                         dim=1, keepdim=True).mean()

        metrics.update({
            "iqlearn_loss_total": loss,
            'entropy': entropy,
        })
        for k, v in iter(metrics.items()):
            metrics[k] = v.item()
        return loss, metrics, final_hidden
