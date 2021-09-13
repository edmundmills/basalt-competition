from agents.base_network import Network
from helpers.environment import ObservationSpace, ActionSpace

import os
import time

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Currently non-functional


class BCAgent:
    def __init__(self):
        self.actions = ActionSpace.actions()
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.model = Network().to(self.device)

    def load_parameters(self, model_file_path):
        self.model.load_state_dict(
            th.load(model_file_path, map_location=self.device), strict=False)

    def get_action(self, trajectory):
        (current_pov, current_inventory, current_equipped,
            frame_sequence) = ObservationSpace.obs_to_state(trajectory.current_obs())
        with th.no_grad():
            probabilities = self.model(current_pov.to(self.device),
                                       current_inventory.to(self.device),
                                       current_equipped.to(self.device),
                                       frame_sequence.to(self.device)).cpu().squeeze()
        probabilities = F.softmax(probabilities, dim=0).numpy()
        action = np.random.choice(self.actions, p=probabilities)
        print(ActionSpace.action_name(action))
        return action

    def train(self, dataset, run):
        optimizer = th.optim.Adam(self.model.parameters(), lr=run.config['learning_rate'])
        dataloader = DataLoader(dataset, batch_size=run.config['batch_size'],
                                shuffle=True, num_workers=4)
        iter_count = 0
        iter_start_time = time.time()
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
        logits = self.model(current_pov.to(self.device),
                            current_inventory.to(self.device),
                            current_equipped.to(self.device),
                            frame_sequence.to(self.device))

        actions = th.from_numpy(actions).long().to(self.device)

        loss = F.cross_entropy(logits, actions)
        return loss
