from helpers.environment import ObservationSpace, ActionSpace
from networks.base_network import Network
from helpers.gpu import states_to_device
import torch as th
import torch.nn.functional as F
import numpy as np

import math
import os


class BC(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def action_probabilities(self, states):
        logits = self.forward(states)
        probabilities = F.softmax(logits, dim=1)
        return probabilities

    def get_action(self, state):
        states = [state_component.unsqueeze(0) for state_component in state]
        states, = states_to_device([states], self.device)
        with th.no_grad():
            Q = self.get_Q(states)
            probabilities = self.action_probabilities(Q).cpu().numpy().squeeze()
        action = np.random.choice(self.actions, p=probabilities)
        return action

    def loss(self, states, actions):
        action_probabilities = self.forward(states)
        actions = actions.squeeze()
        loss = F.cross_entropy(action_probabilities, actions)
        return loss
