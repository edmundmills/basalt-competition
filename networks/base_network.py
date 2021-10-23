from torchvision.models.mobilenetv3 import mobilenet_v3_large

import numpy as np
import torch as th
from torch import nn


def disable_gradients(network):
    for param in network.parameters():
        param.requires_grad = False


class VisualFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_observation_frames = config.n_observation_frames
        self.frame_shape = ObservationSpace.frame_shape
        self.cnn_layers = config.cnn_layers
        mobilenet_features = mobilenet_v3_large(
            pretrained=True, progress=True).features
        if self.n_observation_frames == 1:
            self.cnn = mobilenet_features[0:self.cnn_layers]
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
        self.feature_dim = self._visual_features_dim()

    def forward(self, pov):
        *n, c, h, w = pov.size()
        pov = pov.reshape(-1, c, h, w)
        features = self.cnn(pov)
        return self.cnn(pov).reshape(*n, -1)

    def _visual_features_dim(self):
        with th.no_grad():
            dummy_input = th.zeros((1, 3*self.n_observation_frames, 64, 64))
            output = self.forward(dummy_input)
        print('Base network visual feature dimensions: ', output.size())
        return output.reshape(-1).size()[0]


class LSTMLayer(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.hidden_size = config.lstm_hidden_size
        self.initial_hidden = th.zeros(self.hidden_size * 2)
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=self.hidden_size,
                            num_layers=config.lstm_layers, batch_first=True)

    def forward(self, features, hidden):
        hidden, cell = th.chunk(hidden, 2, dim=-1)
        new_features, new_hidden = self.lstm(features,
                                             (hidden.contiguous(), cell.contiguous()))
        new_hidden = th.cat(new_hidden, dim=-1).squeeze(0).detach().cpu()
        return new_features, new_hidden


class LinearLayers(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_size = config.linear_layer_size
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(input_dim, self.layer_size),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.layer_size, output_dim)
        )

    def forward(self, features):
        *n, feature_dim = features.size()
        out = self.linear(features.reshape(-1, feature_dim)).reshape(*n, -1)
        return out


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_observation_frames = config.n_observation_frames
        self.actions = ActionSpace.actions()
        self.item_dim = 2 * len(ObservationSpace.items())
        self.output_dim = len(self.actions)
        self.visual_feature_extractor = VisualFeatureExtractor(config)
        linear_input_dim = sum([self.visual_feature_extractor.feature_dim,
                                self.item_dim])
        if config.lstm_layers > 0:
            self.lstm = LSTMLayer(linear_input_dim, config)
            linear_input_dim = self.lstm.hidden_size
        else:
            self.lstm = None
        self.linear = LinearLayers(linear_input_dim, self.output_dim, config)
        self.print_model_param_count()
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.to(self.device)

    def initial_hidden(self):
        initial_hidden = self.lstm.initial_hidden if self.lstm else None
        return initial_hidden

    def forward(self, state):
        if self.lstm is not None:
            pov, items, hidden = state
            # only use initial hidden state
            hidden = hidden[:, 0, :].squeeze(dim=1)
            # add dimension D for non-bidirectional
            hidden = hidden.unsqueeze(0)
        else:
            pov, items = state
        visual_features = self.visual_feature_extractor(pov)
        features = th.cat((visual_features, items), dim=-1)
        if self.lstm is not None:
            features, hidden = self.lstm(features, hidden)
        else:
            hidden = None
        return self.linear(features), hidden

    def print_model_param_count(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Number of model params: ', params)

    def load_parameters(self, model_file_path):
        self.load_state_dict(
            th.load(model_file_path, map_location=self.device), strict=False)

    def save(self, path):
        th.save(self.state_dict(), path)
