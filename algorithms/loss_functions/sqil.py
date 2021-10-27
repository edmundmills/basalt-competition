import torch as th
import torch.nn.functional as F


class SQILLoss:
    def __init__(self, model, config):
        self.model = model
        self.actions = ActionSpace.actions()
        self.alpha = config.method.alpha
        self.discount_factor = config.method.discount_factor
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    def __call__(self, expert, policy, expert_no_aug=None, policy_no_aug=None):
        metrics = {}
        expert_states, expert_actions, _expert_replay, expert_next_states, \
            _expert_done = expert
        replay_states, replay_actions, _replay_rewards, replay_next_states, \
            _replay_done,  = policy

        expert_actions = expert_actions.unsqueeze(1)
        replay_actions = replay_actions.unsqueeze(1)

        expert_batch_size = expert_actions.size()[0]
        replay_batch_size = replay_actions.size()[0]
        expert_rewards = th.ones(expert_batch_size, 1, device=self.device)
        replay_rewards = th.ones(replay_batch_size, 1, device=self.device)

        batch_states = cat_states(expert_states, replay_states,
                                  expert_next_states, replay_next_states)

        batch_rewards = th.cat([expert_rewards,
                                replay_rewards], dim=0)

        batch_Qs, final_hidden = self.model.get_Q(batch_states)
        if final_hidden.size()[0] != 0:
            final_hidden, _ = final_hidden.chunk(2, dim=0)

        current_Qs, next_Qs = th.chunk(batch_Qs, 2, dim=0)
        predicted_Qs = th.gather(current_Qs, 1, batch_actions)

        V_next = self.model.get_V(next_Qs)
        target_Qs = batch_rewards + self.discount_factor * V_next

        loss = F.mse_loss(predicted_Qs, target_Qs)

        with th.no_grad():
            entropies = self.model.entropies(current_Qs)
            action_probabilities = self.model.action_probabilities(current_Qs)
        entropy = th.sum(action_probabilities.detach() * entropies.detach(),
                         dim=1, keepdim=True).mean()

        metrics['sqil_loss'] = loss.detach().item()
        metrics['entropy'] = entropy
        return loss, metrics, final_hidden
