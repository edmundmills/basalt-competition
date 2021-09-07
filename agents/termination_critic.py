from helpers.environment import ObservationSpace, ActionSpace
from torchvision.models.mobilenetv3 import mobilenet_v3_large
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
        self.sample_interval = 100
        self.included_steps = self._get_included_steps()

    def _get_included_steps(self):
        included_steps = []
        for idx, step_path in enumerate(self.dataset.step_paths):
            step_dict = self.dataset._load_step_dict(idx)
            equipped_item = step_dict['obs']['equipped_items']['mainhand']['type']
            action = step_dict['action']
            if (step_dict['step'] % self.sample_interval == 0
                    or (equipped_item == 'snowball' and action['use'] == 1)):
                included_steps.append(idx)
        return included_steps

    def __len__(self):
        return len(self.included_steps)

    def __getitem__(self, idx):
        step_idx = self.included_steps[idx]
        return self.dataset[step_idx]


class TerminationCritic():
    self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    self.model = CriticNetwork().to(self.device)

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
        del dataloader

    def loss(self, termination_obs, termination_actions):
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


class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_shape = ObservationSpace.frame_shape
        self.inventory_dim = len(ObservationSpace.items())
        self.equip_dim = len(ObservationSpace.items())
        self.output_dim = len(ActionSpace.actions())
        self.number_of_frames = ObservationSpace.number_of_frames
        self.cnn = mobilenet_v3_large(pretrained=True, progress=True).features
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
            nn.Linear(1024, 1)
        )

    def forward(self, current_pov, current_inventory, current_equipped, frame_sequence):
        batch_size = current_pov.size()[0]
        frame_sequence = frame_sequence.reshape((-1, *self.frame_shape))
        past_visual_features = self.cnn(frame_sequence).reshape(batch_size, -1)
        current_visual_features = self.cnn(current_pov).reshape(batch_size, -1)
        x = th.cat((current_visual_features, current_inventory,
                    current_equipped, past_visual_features), dim=1)
        return th.sigmoid(self.linear(x))

    def _visual_features_shape(self):
        with th.no_grad():
            dummy_input = th.zeros(1, *self.frame_shape)
            output = self.cnn(dummy_input)
        return output.size()

    def _visual_features_dim(self):
        return np.prod(list(self._visual_features_shape()))
