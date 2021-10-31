from networks.base_network import Network

import numpy as np


class Agent(Network):
    def __init__(self, config,):
        super().__init__(config)
        self.termination_confidence_threshhold \
            = config.context.termination_confidence_threshhold \
            if 'termination_confidence_threshhold' in config.context.keys() else 0

    def get_action(self, state):
        return NotImplementedError

    def suppress_unconfident_termination(self, state, action, probabilities):
        if self.context.voluntary_termination and probabilities[self.context.use_action] \
                < self.termination_confidence_threshhold:
            while self.context.termination_helper.terminated(state, action):
                action = np.random.choice(self.actions, p=probabilities)
                print('Tried to terminate_episode, but only had a confidence of',
                      probabilities[self.context.use_action])
        return action
