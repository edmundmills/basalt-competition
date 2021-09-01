from helpers.environment import ObservationSpace, ActionSpace
from torchvision.models.mobilenetv3 import mobilenet_v3_small

import os
import time

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class BCAgent:
    def __init__(self):
        self.actions = ActionSpace.actions()
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.model = BC().to(self.device)

    def load_parameters(self, model_file_path):
        self.model.load_state_dict(
            th.load(model_file_path, map_location=self.device), strict=False)

    def get_action(self, trajectory):
        (current_pov, current_inventory,
         current_equipped, frame_sequence) = trajectory.current_state()
        with th.no_grad():
            probabilities = self.model(current_pov, current_inventory,
                                       current_equipped, frame_sequence).cpu().squeeze()
        probabilities = F.softmax(probabilities).numpy()
        action = np.random.choice(self.actions, p=probabilities)
        return action

    def train(self, dataset, run):
        optimizer = th.optim.Adam(self.model.parameters(), lr=run.lr)
        dataloader = DataLoader(dataset, batch_size=32,
                                shuffle=True, num_workers=4)
        iter_count = 0
        iter_start_time = time.time()
        for epoch in range(run.epochs):
            for _, (dataset_obs, dataset_actions, _done) in enumerate(dataloader):
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


class BC(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_shape = ObservationSpace.frame_shape
        self.inventory_dim = len(ObservationSpace.items().keys())
        self.equip_dim = len(ObservationSpace.items().keys())
        self.output_dim = len(ActionSpace.actions())
        self.number_of_frames = ObservationSpace.number_of_frames
        self.cnn = mobilenet_v3_small(pretrained=True, progress=True).features
        self.visual_feature_dim = self._visual_features_dim()
        self.linear_input_dim = sum([self.visual_feature_dim * self.number_of_frames,
                                     self.inventory_dim,
                                     self.equip_dim])

        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(self.linear_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, self.output_dim)
        )

    def forward(self, current_pov, current_inventory, current_equipped, frame_sequence):
        batch_size = current_pov.size()[0]
        frame_sequence = frame_sequence.reshape((-1, *self.frame_shape))
        past_visual_features = self.cnn(frame_sequence).reshape(batch_size, -1)
        current_visual_features = self.cnn(current_pov).reshape(batch_size, -1)
        x = th.cat((current_visual_features, current_inventory,
                    current_equipped, past_visual_features), dim=1)
        return self.linear(x)

    def _visual_features_shape(self):
        with th.no_grad():
            dummy_input = th.zeros(1, *self.frame_shape)
            output = self.cnn(dummy_input)
        return output.size()

    def _visual_features_dim(self):
        return np.prod(list(self._visual_features_shape()))
