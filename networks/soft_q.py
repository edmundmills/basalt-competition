from helpers.environment import ObservationSpace, ActionSpace
from networks.base_network import Network
from helpers.trajectories import Trajectory
from helpers.datasets import MixedReplayBuffer

import wandb
import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np

import math
import os


class SoftQNetwork(Network):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def get_Q(self, state):
        return self.forward(state)

    def get_Q_s_a(self, states, actions):
        Qs = self.get_Q(states)
        Q_s_a = th.gather(Qs, dim=1, index=actions.unsqueeze(1))
        return Q_s_a

    def get_V(self, Qs):
        v = self.alpha * th.logsumexp(Qs / self.alpha, dim=1, keepdim=True)
        return v

    def action_probabilities(self, Qs):
        probabilities = F.softmax(Qs/self.alpha, dim=1)
        return probabilities

    def entropy(self, Qs):
        entropies = F.log_softmax(Qs/self.alpha, dim=1).sum(dim=1, keepdim=True)
        return entropies

    def get_action(self, state):
        state = [state_component.to(self.device) for state_component in state]
        with th.no_grad():
            Q = self.get_Q(state)
            probabilities = self.action_probabilities(Q).cpu().numpy().squeeze()
        action = np.random.choice(self.actions, p=probabilities)
        return action


class TwinnedSoftQNetwork(nn.Module):
    def __init__(self, alpha, **kwargs):
        super().__init__()
        self.alpha = alpha
        self._q_network_1 = SoftQNetwork(alpha, **kwargs)
        self._q_network_2 = SoftQNetwork(alpha, **kwargs)

    def get_Q(self, state):
        return self._q_network_1.get_Q(state), self._q_network_2.get_Q(state)

    def get_Q_s_a(self, states, actions):
        Q1_s_a = self._q_network_1.get_Q_s_a(states, actions)
        Q2_s_a = self._q_network_2.get_Q_s_a(states, actions)
        return Q1_s_a, Q2_s_a

    def get_V(self, Qs):
        return self._q_network_1.get_V(Qs)
