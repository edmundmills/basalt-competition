from helpers.environment import ObservationSpace, ActionSpace
from torchvision.models.mobilenetv3 import mobilenet_v3_large

import numpy as np
import torch as th
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.actions = ActionSpace.actions()
        self.frame_shape = ObservationSpace.frame_shape
        self.inventory_dim = len(ObservationSpace.items())
        self.equip_dim = len(ObservationSpace.items())
        self.output_dim = len(self.actions)
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
            nn.Linear(1024, self.output_dim)
        )

    def forward(self, state):
        current_pov, current_inventory, current_equipped, frame_sequence = state
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
