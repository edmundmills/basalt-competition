import torch.nn.functional as F


class BCLoss:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def __call__(self, expert, expert_no_aug=None):
        expert_states, expert_actions, _expert_rewards, _expert_next_states, \
            _expert_done = expert

        action_probabilities, final_hidden = self.model.action_probabilities(
            expert_states)

        metrics = {}

        actions = expert_actions.squeeze()
        loss = F.cross_entropy(action_probabilities, actions)

        metrics['Training/loss'] = loss

        for k, v in iter(metrics.items()):
            metrics[k] = v.item()
        return loss, metrics, final_hidden
