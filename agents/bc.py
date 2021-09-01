from helpers.environment import ObservationSpace, ActionSpace
from torchvision.models.mobilenetv3 import mobilenet_v3_small

import os

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class BCAgent:
    def __init__(self, device=th.device('cpu')):
        run_name = os.getenv('RUN_NAME')
        self.agent_name = f'bc_agent_{run_name}'
        self.device = device
        self.actions = ActionSpace.actions()
        self.model = BC().to(device)

    def load_parameters(self, model_file_path):
        self.model.load_state_dict(
            th.load(model_file_path, map_location=self.device), strict=False)

    def get_action(self, observations):
        with th.no_grad():
            probabilities = self.model(observations).cpu().squeeze()
        probabilities = F.softmax(probabilities).numpy()
        action = np.random.choice(self.actions, p=probabilities)
        return action

    def train(self, dataset, epochs, lr):
        optimizer = th.optim.Adam(self.model.parameters(), lr=lr)
        dataloader = DataLoader(dataset, batch_size=32,
                                shuffle=True, num_workers=4)
        iter_count = 0
        losses = []
        smoothed_losses = []

        for epoch in range(epochs):
            for _, (dataset_obs, _, dataset_actions, _) in tqdm(enumerate(
                    dataloader)):
                loss = self.loss(dataset_obs, dataset_actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_count += 1
                losses.append(loss.detach())
                if (iter_count % 20) == 0:
                    smoothed_losses.append(sum(losses)/len(losses))
                    losses.clear()

                if (iter_count % 1000) == 0:
                    mean_loss = sum(
                        smoothed_losses[-50:-1])/len(smoothed_losses[-50:-1])
                    tqdm.write("Iteration {}. Loss {:<10.3f}".format(
                        iter_count, mean_loss))

        print('Training complete')
        th.save(agent.model.state_dict(), os.path.join(
            'train', f'{self.model_name}.pth'))
        del dataloader

    def loss(self, dataset_obs, dataset_actions):
        current_pov = ObservationSpace.dataset_obs_batch_to_pov(dataset_obs)
        current_inventory = ObservationSpace.dataset_obs_batch_to_inventory(dataset_obs)
        current_equipped = ObservationSpace.dataset_obs_batch_to_equipped(dataset_obs)
        frame_sequence = ObservationSpace.dataset_obs_batch_to_frame_sequence(dataset_obs)
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
        logits = self.model(current_pov.to(device),
                            current_inventory.to(device),
                            current_equipped.to(device),
                            frame_sequence.to(device))

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

        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(self.visual_feature_dim * self.number_of_frames +
                      self.inventory_dim, + self.equip_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, self.output_dim)
        )

    def forward(current_pov, current_inventory, current_equipped, frame_sequence):
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
