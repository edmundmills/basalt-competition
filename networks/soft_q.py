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

    def get_V(self, Qs):
        v = self.alpha * th.logsumexp(Qs / self.alpha, dim=1, keepdim=True)
        return v

    def action_probabilities(self, Qs):
        probabilities = F.softmax(Qs/self.alpha, dim=1)
        return probabilities

    def entropies(self, Qs):
        entropies = F.log_softmax(Qs/self.alpha, dim=1)
        return entropies

    def get_action(self, state):
        state = [state_component.to(self.device) for state_component in state]
        with th.no_grad():
            Q = self.get_Q(state)
            probabilities = self.action_probabilities(Q).cpu().numpy().squeeze()
        action = np.random.choice(self.actions, p=probabilities)
        return action

    def random_action(self, obs, surpress_snowball=True):
        action = np.random.choice(self.actions)
        if surpress_snowball:
            while ActionSpace.threw_snowball(obs, action):
                action = np.random.choice(self.actions)
        return action
