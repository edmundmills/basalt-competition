from helpers.environment import ObservationSpace, ActionSpace
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from agents.bc import BCAgent, NoisyBCAgent, BC
from helpers.datasets import MultiFrameDataset

import os
import time

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class TerminateEpisodeDataset(MultiFrameDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.included_steps = self._get_included_steps()

    def _get_included_steps(self):
        included_steps = []
        for idx, step_path in enumerate(self.dataset.step_paths):
            step_dict = self.dataset._load_step_dict(idx)
            equipped_item = step_dict['obs']['equipped_items']['mainhand']['type']
            action = step_dict['action']
            if step_dict['step'] % 100 == 0 or (equipped_item == 'snowball'
                                                and action['use'] == 1):
                included_steps.append(idx)
        return included_steps

    def __len__(self):
        return len(self.included_steps)

    def __getitem__(self, idx):
        step_idx = self.included_steps[idx]
        return self.dataset[step_idx]


class BCXAgent(BCAgent):
    def __init__(self):
        super().__init__()
        self.model = BCX().to(self.device)

    def get_action(self, trajectory):
        (current_pov, current_inventory,
         current_equipped, frame_sequence) = trajectory.current_state()
        with th.no_grad():
            (probabilities,
             terminate_episode) = self.model(current_pov.to(self.device),
                                             current_inventory.to(self.device),
                                             current_equipped.to(self.device),
                                             frame_sequence.to(self.device))
        probabilities = F.softmax(probabilities.squeeze(), dim=0).cpu().numpy()
        action = np.random.choice(self.actions, p=probabilities)
        while ActionSpace.threw_snowball(trajectory.current_obs(), action):
            action = np.random.choice(self.actions, p=probabilities)
            # print('Preemptively attempted to end the episode')
        # print(ActionSpace.action_name(action))
        # print(f'Termination prediction: {terminate_episode.item()}')
        if terminate_episode > 0.5:
            action = 11
        return action

    def train(self, dataset, run):
        optimizer = th.optim.Adam(self.model.parameters(), lr=run.lr)
        dataloader = DataLoader(dataset, batch_size=32,
                                shuffle=True, num_workers=4)
        termination_dataset = TerminateEpisodeDataset(dataset)
        termination_data_iter = iter(DataLoader(termination_dataset,
                                                batch_size=16, shuffle=True,
                                                num_workers=4, drop_last=True))
        iter_count = 0
        iter_start_time = time.time()
        for epoch in range(run.epochs):
            for _, (dataset_obs, dataset_actions,
                    _next_obs, _done) in enumerate(dataloader):
                loss = self.loss(dataset_obs, dataset_actions)

                (termination_obs, termination_actions, _termination_next_obs,
                 _termination_done) = self._sample_termination_iter(termination_data_iter,
                                                                    termination_dataset)
                loss += self.termination_loss(termination_obs, termination_actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_count += 1
                run.append_loss(loss.detach().cpu())
                run.print_update(iter_count)

        print('Training complete')
        th.save(self.model.state_dict(), os.path.join('train', f'{run.name}.pth'))
        run.save_data()
        del dataloader

    def loss(self, dataset_obs, dataset_actions):
        current_pov = ObservationSpace.obs_to_pov(dataset_obs)
        current_inventory = ObservationSpace.obs_to_inventory(dataset_obs)
        current_equipped = ObservationSpace.obs_to_equipped_item(dataset_obs)
        frame_sequence = ObservationSpace.obs_to_frame_sequence(dataset_obs)
        actions = ActionSpace.dataset_action_batch_to_actions(dataset_actions)

        # Remove samples that had no corresponding action
        mask = actions != -1
        current_pov = current_pov[mask]
        current_inventory = current_inventory[mask]
        frame_sequence = frame_sequence[mask]
        current_equipped = current_equipped[mask]
        actions = actions[mask]

        if len(actions) == 0:
            return 0

        # Obtain logits of each action
        action_logits, _terminate_episode = self.model(current_pov.to(self.device),
                                                       current_inventory.to(self.device),
                                                       current_equipped.to(self.device),
                                                       frame_sequence.to(self.device))

        actions = th.from_numpy(actions).long().to(self.device)

        loss = F.cross_entropy(action_logits, actions)
        return loss

    def _sample_termination_iter(self, termination_data_iter, termination_dataset):
        try:
            (termination_obs, termination_actions, _termination_next_obs,
             _termination_done) = next(termination_data_iter)
        except StopIteration:
            termination_data_iter = iter(DataLoader(termination_dataset,
                                                    batch_size=16,
                                                    shuffle=True, num_workers=4))
            (termination_obs, termination_actions, _termination_next_obs,
             _termination_done) = next(termination_data_iter)
        return (termination_obs, termination_actions,
                _termination_next_obs, _termination_done)

    def termination_loss(self, termination_obs, termination_actions):
        current_pov = ObservationSpace.obs_to_pov(termination_obs)
        current_inventory = ObservationSpace.obs_to_inventory(termination_obs)
        current_equipped = ObservationSpace.obs_to_equipped_item(termination_obs)
        frame_sequence = ObservationSpace.obs_to_frame_sequence(termination_obs)
        actions = ActionSpace.dataset_action_batch_to_actions(termination_actions)

        use_actions = th.from_numpy(actions == 11).unsqueeze(1)
        snowball_equipped = current_equipped == ActionSpace.one_hot_snowball()
        terminated = use_actions * snowball_equipped

        _action_logits, predict_terminate = self.model(current_pov.to(self.device),
                                                       current_inventory.to(self.device),
                                                       current_equipped.to(self.device),
                                                       frame_sequence.to(self.device))

        loss = F.binary_cross_entropy(predict_terminate,
                                      terminated.float().to(self.device))
        return loss


class NoisyBCXAgent(BCXAgent):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        super().__init__()

    def get_action(self, trajectory):
        if np.random.rand() < self.epsilon / 1000:
            action = 11
            # print(f'Threw Snowball (at random)')
        elif np.random.rand() < self.epsilon:
            action = np.random.choice(ActionSpace.actions())
            while ActionSpace.threw_snowball(trajectory.current_obs(), action):
                action = np.random.choice(self.actions)
            # print(f'{ActionSpace.action_name(action)} (at random)')
        else:
            action = super().get_action(trajectory)
        return action


class BCX(BC):
    def __init__(self):
        super().__init__()
        self.output_dim = len(ActionSpace.actions()) + 1

        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(self.linear_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, self.output_dim)
        )

    def forward(self, current_pov, current_inventory, current_equipped, frame_sequence):
        output = super().forward(current_pov, current_inventory,
                                 current_equipped, frame_sequence)
        action_values, terminate_episode = output.split([len(ActionSpace.actions()), 1],
                                                        dim=1)
        return action_values, th.sigmoid(terminate_episode)
