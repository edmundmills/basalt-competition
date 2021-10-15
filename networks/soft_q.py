from networks.base_network import Network
from networks.termination_critic import TerminationCritic
from utils.environment import ActionSpace

import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np
import wandb


class SoftQNetwork(Network):
    def __init__(self, config):
        super().__init__(config)
        self.alpha = config.alpha
        self.termination_critic = TerminationCritic(config) \
            if config.env.termination_critic else None
        self.termination_confidence_threshhold = config.termination_confidence_threshhold

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

    def get_action(self, states, iter_count=None):
        with th.no_grad():
            Q, hidden = self.get_Q(states)
            probabilities = self.action_probabilities(Q).cpu().numpy().squeeze()
        action = np.random.choice(self.actions, p=probabilities)
        threw_snowball = ActionSpace.threw_snowball(states, action, device=self.device)
        if self.config.wandb and len(probabilities) >= 12:
            wandb.log({'TerminationCritic/use_action_prob': probabilities[11]},
                      step=iter_count)
        if self.termination_critic is not None:
            while threw_snowball:
                action = np.random.choice(self.actions, p=probabilities)
                threw_snowball = ActionSpace.threw_snowball(states, action,
                                                            device=self.device)
            if iter_count is None or iter_count % 10 == 0:
                eval = self.termination_critic.evaluate(states)
                if self.config.wandb:
                    wandb.log({'TerminationCritic/state_eval': eval},
                              step=iter_count)
                if eval > self.termination_confidence_threshhold:
                    print('termination_critic:', eval)
                    if ActionSpace.snowball_equipped(states, device=self.device):
                        action = ActionSpace.use_action()
                        print("Snowball thrown by termination_critic")
                    else:
                        action = ActionSpace.equip_snowball_action()
                        print("Snowball equipped by termination_critic")
        elif len(probabilities) >= 12 \
                and probabilities[11] < self.termination_confidence_threshhold:
            while threw_snowball:
                action = np.random.choice(self.actions, p=probabilities)
                threw_snowball = ActionSpace.threw_snowball(states, action,
                                                            device=self.device)
                print('Tried to throw snowball, but only had a confidence of',
                      probabilities[11])

        if hidden is not None:
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
