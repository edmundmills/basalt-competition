from networks.base_network import Network
from algorithms.offline import SupervisedLearning
from core.gpu import GPULoader

import os
import time
from pathlib import Path

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class TerminateEpisodeDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.sample_interval = 100
        self.included_steps = self._get_included_steps()

    def _get_included_steps(self):
        included_steps = []
        for trajectory_idx, trajectory in enumerate(self.dataset.trajectories):
            trajectory_length = len(trajectory)
            for step_idx, (state, action, _, _, _) in enumerate(trajectory):
                if (step_idx % self.sample_interval == 0
                    and (step_idx < trajectory_length - self.sample_interval)
                        or ActionSpace.threw_snowball(state, action)):
                    included_steps.append((trajectory_idx, step_idx))
        return included_steps

    def __len__(self):
        return len(self.included_steps)

    def __getitem__(self, idx):
        trajectory_idx, step_idx = self.included_steps[idx]
        return self.dataset.trajectories[trajectory_idx][step_idx]


class TerminationCritic(Network):
    def __init__(self, config):
        super().__init__(config)
        self.lstm = None
        self.linear_input_dim = sum([self.visual_feature_extractor.feature_dim,
                                     self.item_dim])
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.linear_input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        self.to(self.device)

    def forward(self, state):
        pov = state[0]
        items = state[1]
        visual_features = self.visual_feature_extractor(pov)
        features = th.cat((visual_features, items), dim=-1)
        return th.sigmoid(self.linear(features))

    def evaluate(self, state):
        with th.no_grad():
            evaluation = self.forward(state)
        return evaluation.item()

    def train(self, dataset):
        termination_dataset = TerminateEpisodeDataset(dataset)
        training_algorithm = SupervisedLearning(
            self.config, epochs=self.config.env.termination_critic_training_epochs)
        training_algorithm(self, termination_dataset)
        self.steps_trained = training_algorithm.iter_count

    def loss(self, states, actions):
        threw_snowball = ActionSpace.threw_snowball_tensor(states, actions, self.device)
        predict_terminate = self.forward(states)
        loss = F.binary_cross_entropy(predict_terminate, threw_snowball.float())
        return loss

    def save(self, path=None):
        th.save(self.state_dict(), os.path.join(
            'train', f'{self.config.env.name}_termination_critic.pth'))

    def critique_trajectory(self, trajectory):
        print('Critiquing Trajectory')
        gpu_loader = GPULoader(self.config)
        gpu_loader.loading_sequences = False
        for step, state in enumerate(trajectory.states):
            state = gpu_loader.state_to_device(state)
            eval = self.evaluate(state)
            print(eval)
        trajectory.save_as_video(Path('eval'), 'critiqued_trajectory')
        print(len(trajectory))
        print('Critiquing Complete')
