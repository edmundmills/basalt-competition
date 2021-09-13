from helpers.environment import ObservationSpace, ActionSpace
from agents.base_network import Network

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
                                     self.inventory_dim,
                                     self.equip_dim])
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.linear_input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, current_pov, current_inventory, current_equipped):
        current_visual_features = self.cnn(current_pov).flatten(start_dim=1)
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
            obs = trajectory.get_obs(step, n_observation_frames=1)
            (current_pov, current_inventory,
                current_equipped) = ObservationSpace.obs_to_state(obs, device=self.device)
            with th.no_grad():
                rating = self.model.forward(current_pov, current_inventory,
                                            current_equipped)
                print(rating.item())
            termination_ratings.append(rating.item())
            reward = self.termination_reward(state)
            print(reward)
            termination_rewards.append(reward)
        trajectory.additional_data['termination_ratings'] = termination_ratings
        trajectory.additional_data['termination_rewards'] = termination_rewards
        return termination_ratings

    def termination_reward(self, state):
        state = [state_component.to(self.device) for state_component in state]
        if len(state) == 4:
            current_pov, current_inventory, current_equipped, frame_sequence = state
            frames = (*th.unbind(frame_sequence, dim=1), current_pov)
        else:
            current_pov, current_inventory, current_equipped = state
            frames = [current_pov]
        with th.no_grad():
            ratings = [self.model(frame, current_inventory, current_equipped).item()
                       for frame in frames]
        average_rating = sum(ratings) / len(ratings)
        reward = min((average_rating * 20000) - 1, 2.0)
        return reward

    def train(self, dataset, run):
        optimizer = th.optim.Adam(self.model.parameters(), lr=run.config['learning_rate'])
        termination_dataset = TerminateEpisodeDataset(dataset)
        dataloader = DataLoader(termination_dataset, batch_size=run.config['batch_size'],
                                shuffle=True, num_workers=4)

        iter_count = 0
        for epoch in range(run.config['epochs']):
            for _, (dataset_obs, dataset_actions,
                    _next_obs, _done) in enumerate(dataloader):
                loss = self.loss(dataset_obs, dataset_actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_count += 1
                run.append_loss(loss.detach().cpu())
                run.print_update(iter_count)
            print(f'Epoch #{epoch + 1} completed')

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
        batch_size = use_actions.size()[0]
        snowball_tensor = ActionSpace.one_hot_snowball().repeat(batch_size, 1)
        snowball_equipped = th.all(
            th.eq(current_equipped, snowball_tensor), dim=1, keepdim=True)
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
