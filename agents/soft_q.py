from agents.base import Agent

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F


class SoftQAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.alpha = config.method.alpha

    def get_Q(self, state):
        return self.forward(state)

    def get_Q_s_a(self, states, actions):
        Qs, hidden = self.get_Q(states)
        Q_s_a = th.gather(Qs, dim=-1, index=actions.reshape(-1, 1))
        return Q_s_a, hidden

    def get_V(self, Qs):
        v = self.alpha * th.logsumexp(Qs / self.alpha, dim=-1, keepdim=True)
        return v

    def action_probabilities(self, Qs):
        probabilities = F.softmax(Qs/self.alpha, dim=-1)
        return probabilities

    def entropies(self, Qs):
        entropies = -F.log_softmax(Qs/self.alpha, dim=-1)
        return entropies

    def batch_entropy(self, Qs):
        Qs = Qs.reshape(-1, len(self.actions))
        action_probabilities = self.action_probabilities(Qs).mean(dim=0, keepdim=False)
        entropy = -th.sum(action_probabilities * th.log(action_probabilities))
        return entropy

    def get_action(self, state):
        with th.no_grad():
            Q, hidden = self.get_Q(state)
            probabilities = self.action_probabilities(Q).cpu().numpy().squeeze()
        action = np.random.choice(self.actions, p=probabilities)
        action = self.suppress_unconfident_termination(state, action, probabilities)
        hidden = hidden.cpu().squeeze()
        return action, hidden

    def save(self, path):
        state_dict = self.state_dict()
        state_dict['alpha'] = self.alpha
        th.save(state_dict, path)

    def load_parameters(self, model_file_path):
        state_dict = th.load(model_file_path, map_location=self.device)
        if 'alpha' in list(state_dict.keys()):
            self.alpha = state_dict['alpha']
        self.load_state_dict(state_dict, strict=False)


class TwinnedSoftQAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha = config.method.alpha
        self._q_network_1 = SoftQAgent(config)
        self._q_network_2 = SoftQAgent(config)

    def get_Q(self, state):
        return self._q_network_1.get_Q(state), self._q_network_2.get_Q(state)

    def get_Q_s_a(self, states, actions):
        Q1_s_a = self._q_network_1.get_Q_s_a(states, actions)
        Q2_s_a = self._q_network_2.get_Q_s_a(states, actions)
        return Q1_s_a, Q2_s_a

    def get_V(self, Qs):
        return self._q_network_1.get_V(Qs)
