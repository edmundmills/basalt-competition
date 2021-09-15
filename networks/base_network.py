from helpers.environment import ObservationSpace, ActionSpace
from torchvision.models.mobilenetv3 import mobilenet_v3_large

import numpy as np
import torch as th
from torch import nn


class Network(nn.Module):
    def __init__(self, n_observation_frames=1):
        super().__init__()
        self.n_observation_frames = n_observation_frames
        self.actions = ActionSpace.actions()
        self.frame_shape = ObservationSpace.frame_shape
        self.item_dim = 2 * len(ObservationSpace.items())
        self.output_dim = len(self.actions)
        mobilenet_features = mobilenet_v3_large(
            pretrained=True, progress=True).features
        if self.n_observation_frames == 1:
            self.cnn = mobilenet_features
        else:
            self.cnn = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(3*self.n_observation_frames, 16, kernel_size=(3, 3),
                              stride=(2, 2), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(16, eps=0.001, momentum=0.01,
                                   affine=True, track_running_stats=True),
                    nn.Hardswish()),
                *mobilenet_features[1:]
            )
        self.visual_feature_dim = self._visual_features_dim()
        self.linear_input_dim = sum([self.visual_feature_dim,
                                     self.item_dim])

        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(self.linear_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, self.output_dim)
        )
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        pov, items = state
        batch_size = pov.size()[0]
        visual_features = self.cnn(pov).reshape(batch_size, -1)
        x = th.cat((visual_features, items), dim=1)
        return self.linear(x)

    def _visual_features_shape(self):
        with th.no_grad():
            dummy_input = th.zeros((1, 3*self.n_observation_frames, 64, 64))
            output = self.cnn(dummy_input)
        return output.size()

    def _visual_features_dim(self):
        return np.prod(list(self._visual_features_shape()))

    def load_parameters(self, model_file_path):
        self.load_state_dict(
            th.load(model_file_path, map_location=self.device), strict=False)

    def save(self, path):
        th.save(self.state_dict(), path)
