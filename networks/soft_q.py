from networks.base_network import Network

import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np


class SoftQNetwork(Network):
    def __init__(self, config):
        super().__init__(config)
        self.alpha = config.alpha

    def get_Q(self, state):
        return self.forward(state)

    def get_Q_s_a(self, states, actions):
        Qs, hidden = self.get_Q(states)
        Q_s_a = th.gather(Qs, dim=1, index=actions.reshape(-1, 1))
        return Q_s_a, hidden

    def get_V(self, Qs):
        v = self.alpha * th.logsumexp(Qs / self.alpha, dim=1, keepdim=True)
        return v

    def action_probabilities(self, Qs):
        probabilities = F.softmax(Qs/self.alpha, dim=1)
        return probabilities

    def entropies(self, Qs):
        entropies = -F.log_softmax(Qs/self.alpha, dim=1)
        return entropies

    def get_action(self, state):
        states = [state_component.unsqueeze(0) for state_component in state]
        states, = self.gpu_loader.states_to_device([states])
        with th.no_grad():
            Q, hidden = self.get_Q(states)
            probabilities = self.action_probabilities(Q).cpu().numpy().squeeze()
        action = np.random.choice(self.actions, p=probabilities)
        return action, hidden


class TwinnedSoftQNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alpha = config.alpha
        self._q_network_1 = SoftQNetwork(config)
        self._q_network_2 = SoftQNetwork(config)

    def get_Q(self, state):
        return self._q_network_1.get_Q(state), self._q_network_2.get_Q(state)

    def get_Q_s_a(self, states, actions):
        Q1_s_a = self._q_network_1.get_Q_s_a(states, actions)
        Q2_s_a = self._q_network_2.get_Q_s_a(states, actions)
        return Q1_s_a, Q2_s_a

    def get_V(self, Qs):
        return self._q_network_1.get_V(Qs)
