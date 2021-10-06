from utils.environment import ObservationSpace, ActionSpace
from networks.base_network import Network

import os
import time

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
            for step_idx, (obs, action, _, _) in enumerate(trajectory):
                equipped_item = obs['equipped_items']['mainhand']['type']
                if (step_idx % self.sample_interval == 0
                    and (step_idx < trajectory_length - self.sample_interval)
                        or (equipped_item == 'snowball' and action['use'] == 1)):
                    included_steps.append((trajectory_idx, step_idx))
        return included_steps

    def __len__(self):
        return len(self.included_steps)

    def __getitem__(self, idx):
        trajectory_idx, step_idx = self.included_steps[idx]
        return self.dataset.trajectories[trajectory_idx][step_idx]


class CriticNetwork(Network):
    def __init__(self):
        super().__init__()
        self.linear_input_dim = sum([self.visual_feature_dim,
                                     self.item_dim])
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.linear_input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, state):
        pov, items = state
        current_visual_features = self.cnn(pov).flatten(start_dim=1)
        x = th.cat((current_visual_features, items), dim=1)
        return th.sigmoid(self.linear(x))


class TerminationCritic():
    def __init__(self):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.model = CriticNetwork().to(self.device)

    def critique_trajectory(self, trajectory):
        termination_ratings = []
        termination_rewards = []
        for step in range(len(trajectory)):
            obs = trajectory.get_obs(step, n_observation_frames=1)
            state = ObservationSpace.obs_to_state(obs)
            state = [state_component.to(self.device) for state_component in state]
            with th.no_grad():
                rating = self.model.forward(state)
                print(rating.item())
            termination_ratings.append(rating.item())
            reward = self.termination_reward(state)
            print(reward)
            termination_rewards.append(reward)
        trajectory.additional_data['termination_ratings'] = termination_ratings
        trajectory.additional_data['termination_rewards'] = termination_rewards
        return termination_ratings

    def termination_reward(self, state):
        pov, items = [state_component.to(self.device) for state_component in state]
        frames = th.split(pov, 3, dim=1)
        with th.no_grad():
            ratings = [self.model((frame, items)).item()
                       for frame in frames]
        average_rating = sum(ratings) / len(ratings)
        reward = min((average_rating * 20000) - 1, 2.0)
        return reward

    def train(self, dataset, config):
        termination_dataset = TerminateEpisodeDataset(dataset)
        training_algorithm = SupervisedLearning(config)
        training_algorithm(self, termination_dataset)
        th.save(self.model.state_dict(), os.path.join(
            'train', f'{training_algorithm.name}.pth'))

    def loss(self, termination_obs, termination_actions):
        pov, items = ObservationSpace.obs_to_state(termination_obs)
        actions = ActionSpace.dataset_action_batch_to_actions(termination_actions)

        use_actions = th.from_numpy(actions == 11).unsqueeze(1)
        batch_size = use_actions.size()[0]
        snowball_tensor = ActionSpace.one_hot_snowball().repeat(batch_size, 1)
        snowball_equipped = th.all(
            th.eq(th.chunk(items, 2, dim=1)[1], snowball_tensor), dim=1, keepdim=True)
        terminated = use_actions * snowball_equipped

        predict_terminate = self.model((pov.to(self.device),
                                       items.to(self.device)))

        loss = F.binary_cross_entropy(predict_terminate,
                                      terminated.float().to(self.device))
        return loss

    def parameters(self):
        self.model.parameters()

    def load_parameters(self, model_file_path):
        self.model.load_state_dict(
            th.load(model_file_path, map_location=self.device), strict=False)
