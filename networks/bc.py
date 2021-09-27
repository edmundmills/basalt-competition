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
        state = [state_component.to(self.device) for state_component in state]
        with th.no_grad():
            Q = self.get_Q(state)
            probabilities = self.action_probabilities(Q).cpu().numpy().squeeze()
        action = np.random.choice(self.actions, p=probabilities)
        return action

    def loss(self, states, actions):
        actions = ActionSpace.dataset_action_batch_to_actions(actions)
        mask = actions != -1
        actions = actions[mask]
        actions = th.from_numpy(actions).long().to(self.device)
        states = [state_component[mask].to(self.device)
                  for state_component in states]
        action_probabilities = self.forward(states)
        loss = F.cross_entropy(action_probabilities, actions)
        return loss
