from helpers.environment import ObservationSpace, ActionSpace
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from helpers.gpu import GPULoader

import numpy as np
import torch as th
from torch import nn


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_observation_frames = config.n_observation_frames
        self.cnn_layers = config.cnn_layers
        self.linear_layer_size = config.linear_layer_size
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
                *mobilenet_features[1:self.cnn_layers]
            )
        self.visual_feature_dim = self._visual_features_dim()
        self.linear_input_dim = sum([self.visual_feature_dim,
                                     self.item_dim])

        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(self.linear_input_dim, self.linear_layer_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.linear_layer_size, self.output_dim)
        )
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Number of model params: ', params)
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.to(self.device)
        self.gpu_loader = GPULoader()

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
        print('Base network visual feature shape: ', output.size())
        return output.size()

    def _visual_features_dim(self):
        return np.prod(list(self._visual_features_shape()))

    def load_parameters(self, model_file_path):
        self.load_state_dict(
            th.load(model_file_path, map_location=self.device), strict=False)

    def save(self, path):
        th.save(self.state_dict(), path)
