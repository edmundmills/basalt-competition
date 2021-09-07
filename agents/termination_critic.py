from helpers.environment import ObservationSpace, ActionSpace
from helpers.datasets import StepDataset
from agents.base_network import Network

import os
import time

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class TerminateEpisodeDataset(StepDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.sample_interval = 100
        self.included_steps = self._get_included_steps()

    def _get_included_steps(self):
        included_steps = []
        for idx, step_path in enumerate(self.dataset.step_paths):
            trajectory_length = self.trajectory_length(step_path)
            step_dict = self.dataset._load_step_dict(idx)
            equipped_item = step_dict['obs']['equipped_items']['mainhand']['type']
            action = step_dict['action']
            if ((step_dict['step'] % self.sample_interval == 0
                 and step_dict['step'] < trajectory_length - self.sample_interval)
                    or (equipped_item == 'snowball' and action['use'] == 1)):
                included_steps.append(idx)
        return included_steps

    def __len__(self):
        return len(self.included_steps)

    def __getitem__(self, idx):
        step_idx = self.included_steps[idx]
        return self.dataset[step_idx]


class CriticNetwork(Network):
    def __init__(self):
        super().__init__()
        self.linear_input_dim = sum([self.visual_feature_dim,
                                     self.inventory_dim,
                                     self.equip_dim])
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(self.linear_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1)
        )

    def forward(self, current_pov, current_inventory, current_equipped):
        batch_size = current_pov.size()[0]
        current_visual_features = self.cnn(current_pov).reshape(batch_size, -1)
        x = th.cat((current_visual_features, current_inventory, current_equipped), dim=1)
        return th.sigmoid(self.linear(x))


class TerminationCritic():
    def __init__(self):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.model = CriticNetwork().to(self.device)

    def critique_trajectory(self, trajectory):
        termination_ratings = []
        termination_rewards = []
        for step in range(len(trajectory)):
            (current_pov, current_inventory,
             current_equipped, frame_sequence) = trajectory.get_state(step)
            with th.no_grad():
                rating = self.model.forward(current_pov, current_inventory,
                                            current_equipped)
                print(rating.item())
            termination_ratings.append(rating.detach().cpu())
            reward = self.termination_reward(current_pov, current_inventory,
                                             current_equipped, frame_sequence)
            termination_rewards.append(reward)
        trajectory.additional_data['termination_ratings'] = termination_ratings
        trajectory.additional_data['termination_rewards'] = termination_rewards
        return termination_ratings

    def termination_reward(self, current_pov, current_inventory,
                           current_equipped, frame_sequence):
        frames = frame_sequence.chunk(ObservationSpace.number_of_frames - 1, dim=0)
        with th.no_grad():
            ratings = [self.model(frame, current_inventory, current_equipped).detach()
                       for frame in frames]
            ratings.append(self.model(current_pov, current_inventory,
                                      current_equipped).detach())
        average_rating = sum(ratings) / len(ratings)
        reward = (average_rating * 4000) - 1
        return reward

    def train(self, dataset, run):
        optimizer = th.optim.Adam(self.model.parameters(), lr=run.lr)
        termination_dataset = TerminateEpisodeDataset(dataset)
        dataloader = DataLoader(termination_dataset, batch_size=32,
                                shuffle=True, num_workers=4)

        iter_count = 0
        for epoch in range(run.epochs):
            for _, (dataset_obs, dataset_actions,
                    _next_obs, _done) in enumerate(dataloader):
                loss = self.loss(dataset_obs, dataset_actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_count += 1
                run.append_loss(loss.detach().cpu())
                run.print_update(iter_count)

        print('Training complete')
        th.save(self.model.state_dict(), os.path.join('train', f'{run.name}.pth'))
        run.save_data()
        del termination_dataset
        del dataloader

    def loss(self, termination_obs, termination_actions):
        current_pov = ObservationSpace.obs_to_pov(termination_obs)
        current_inventory = ObservationSpace.obs_to_inventory(termination_obs)
        current_equipped = ObservationSpace.obs_to_equipped_item(termination_obs)
        actions = ActionSpace.dataset_action_batch_to_actions(termination_actions)

        use_actions = th.from_numpy(actions == 11).unsqueeze(1)
        snowball_equipped = current_equipped == ActionSpace.one_hot_snowball()
        terminated = use_actions * snowball_equipped

        predict_terminate = self.model(current_pov.to(self.device),
                                       current_inventory.to(self.device),
                                       current_equipped.to(self.device))

        loss = F.binary_cross_entropy(predict_terminate,
                                      terminated.float().to(self.device))
        return loss

    def load_parameters(self, model_file_path):
        self.model.load_state_dict(
            th.load(model_file_path, map_location=self.device), strict=False)
