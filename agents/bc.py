from agents.base import Agent

import numpy as np
import torch as th
import torch.nn.functional as F


class BCAgent(Agent):
    def __init__(self, config,):
        super().__init__(config)

    def action_probabilities(self, states):
        logits, hidden = self.forward(states)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities, hidden

    def get_action(self, state):
        with th.no_grad():
            probabilities = self.action_probabilities(state).cpu().numpy().squeeze()
        action = np.random.choice(self.actions, p=probabilities)
        action = self.suppress_unconfident_termination(state, action, probabilities)
        return action
