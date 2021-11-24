from core.environment import create_context

import numpy as np
import torch as th
from torch import nn
from torchvision.models.mobilenetv3 import mobilenet_v3_large


class VisualFeatureExtractor(nn.Module):
    """
    Converts an image tensor into an abstract feature space.

    Input tensor must have the dimensions (sample, (optional sequence dim), channels,
    width, height). Channels covers three color dimensions * the number of stacked frames.

    Number of observation frames and number of cnn layers are specified through
    config. Uses a pretrained MobileNetv3 for the cnn layers.
    
    The size of the abstract feature dimension varies with the number of
    cnn_layers, not always in intuitive ways. The _visual_features_dim function returns
    the size of the feature space. The returned tensor has dimensions
    (sample, (optional sequence dim), features).
    """
    def __init__(self, config):
        super().__init__()
        context = create_context(config)
        self.n_observation_frames = config.model.n_observation_frames
        self.frame_shape = context.frame_shape
        self.cnn_layers = config.model.cnn_layers
        mobilenet_features = mobilenet_v3_large(pretrained=True, progress=True).features
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

    def forward(self, spatial):
        *n, c, h, w = spatial.size()
        spatial = spatial.reshape(-1, c, h, w)
        return self.cnn(spatial).reshape(*n, -1)

    def _visual_features_dim(self):
        with th.no_grad():
            dummy_input = th.zeros((1, 3*self.n_observation_frames, 64, 64))
            output = self.forward(dummy_input)
        print('Base network visual feature dimensions: ', output.size())
        return output.reshape(-1).size()[0]


class LSTMLayer(nn.Module):
    """
    Converts features and a hidden state to updated features and next hidden state.

    Input feature state is a concatenation of extracted visual features and the
    nonspacial component of a state. In the feature space, batch precedes the sequence
    dimension.

    Output features have dimension equal to the LSTM hidden size.
    """
    def __init__(self, input_dim, config):
        super().__init__()
        self.hidden_size = config.model.lstm_hidden_size
        self.initial_hidden = th.zeros(self.hidden_size * 2)
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=self.hidden_size,
                            num_layers=config.model.lstm_layers, batch_first=True)

    def forward(self, features, hidden):
        hidden, cell = th.chunk(hidden, 2, dim=-1)
        new_features, new_hidden = self.lstm(features,
                                             (hidden.contiguous(), cell.contiguous()))
        new_hidden = th.cat(new_hidden, dim=-1).squeeze(0).detach().cpu()
        return new_features, new_hidden


class LinearLayers(nn.Module):
    """Takes features and returns values with the specified output dimension size."""

    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_size = config.model.linear_layer_size
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
    """
    The base module that agents inherit from.

    Takes as input a spacial observation, nonspatial observation, and possible hidden
    state and returns a set of values with dimension equal to the dimension of the
    discrete action space. Also returns the next hidden state.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.context = create_context(config)
        self.n_observation_frames = config.model.n_observation_frames
        self.actions = self.context.actions
        self.nonspatial_size = self.context.nonspatial_size
        self.output_dim = len(self.actions)
        self.visual_feature_extractor = VisualFeatureExtractor(config)
        linear_input_dim = sum([self.visual_feature_extractor.feature_dim,
                                self.nonspatial_size])
        if config.model.lstm_layers > 0:
            self.lstm = LSTMLayer(linear_input_dim, config)
            linear_input_dim = self.lstm.hidden_size
        else:
            self.lstm = None
        self.linear = LinearLayers(linear_input_dim, self.output_dim, config)
        self.print_model_param_count()
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.to(self.device)

    def initial_hidden(self):
        initial_hidden = self.lstm.initial_hidden if self.lstm else th.zeros(0)
        return initial_hidden

    def forward(self, state):
        spatial, nonspatial, hidden = state
        visual_features = self.visual_feature_extractor(spatial)
        features = th.cat((visual_features, nonspatial), dim=-1)
        if self.lstm is not None:
            # only use initial hidden state
            hidden = hidden[:, 0, :].squeeze(dim=1)
            # add dimension D for non-bidirectional
            hidden = hidden.unsqueeze(0)
            features, hidden = self.lstm(features, hidden)
        else:
            hidden = th.zeros(0)
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
