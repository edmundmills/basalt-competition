from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from core.environment import create_context
from core.gpu import GPULoader
from core.state import Transition, cat_states

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, n_observation_frames, cnn_layers):
        super().__init__()
        self.n_observation_frames = n_observation_frames
        mobilenet_features = mobilenet_v3_small(pretrained=True, progress=True).features
        self.model = nn.Sequential(
            nn.Sequential(nn.Conv2d(3*n_observation_frames, 16, kernel_size=(3, 3),
                                    stride=(2, 2), padding=(1, 1), bias=False),
                          nn.BatchNorm2d(16, eps=0.001, momentum=0.01,
                                         affine=True, track_running_stats=True),
                          nn.Hardswish()),
            *nn.Sequential(*mobilenet_features[1:cnn_layers]),
            nn.AvgPool2d(2)
        )

    def forward(self, state):
        spatial, nonspatial, hidden = state
        visual_features = self.model(spatial).flatten(start_dim=-1)
        return th.cat(visual_features, nonspatial, dim=-1)

    def _visual_features_shape(self):
        with th.no_grad():
            dummy_input = th.zeros((1, 3*self.n_observation_frames, 64, 64))
            dummy_features = self.features(dummy_input)
        print('Curiosity Module Feature Shape: ', dummy_features.size())
        return dummy_features.size()

    def _visual_features_dim(self, *args):
        return np.prod(list(self._visual_features_shape(*args)))


class InverseDynamicsModel(nn.Module):
    def __init__(self, feature_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, features):
        return self.model(features)


class ForwardDynamicsModel(nn.Module):
    def __init__(self, feature_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, current_features, next_features):
        combined_features = th.cat((current_features, next_features), dim=-1)
        return self.model(combined_features)


class CuriosityModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        context = create_context(config)
        self.gpu_loader = GPULoader(config)

        self.n_observation_frames = config.model.n_observation_frames
        # scales the returned reward
        self.eta = 0.5
        # controls the relative weight of the two different loss loss_functions
        # low values prioritize deducing actions over predicting next features
        self.beta = 0.1
        self.actions = context.actions
        self.frame_shape = context.frame_shape
        self.cnn_layers = config.method.curiosity_cnn_layers
        self.features = FeatureExtractor(self.n_observation_frames, self.cnn_layers)
        self.feature_dim = self.features._visual_features_dim() + context.nonspatial_size
        self.inverse_dynamics = InverseDynamicsModel(self.feature_dim, len(self.actions))
        self.forward_dynamics = ForwardDynamicsModel(self.feature_dim, len(self.actions))

    def predict_action(self, state, next_state):
        states, _ = cat_states((state, next_state))
        features = self.features(states)
        current_features, next_features = th.chunk(features, 2, dim=0)
        action = self.inverse_dynamics(current_features, next_features)
        return action

    def predict_next_features(self, features, action):
        one_hot_actions = F.one_hot(action, len(self.actions))
        x = th.cat((features, one_hot_actions), dim=-1)
        return self.forward_dynamics(x)

    def reward(self, batch, return_transition=False):
        state, actions, _reward, next_state, _done = batch
        states, _ = cat_states((state, next_state))
        with th.no_grad():
            features = self.get_features(states)
            current_features, next_features = th.chunk(features, 2, dim=0)
            predicted_next_features = self.predict_next_features(current_features,
                                                                 actions)
            reward = F.mse_loss(next_features, predicted_next_features, reduction='none')
            reward = th.mean(reward, dim=-1, keepdim=True)
        if return_transition:
            return Transition(state, actions, reward, next_state, _done)
        return reward.squeeze().tolist()

    def loss(self, batch):
        states, actions, _rewards, next_states, _done = batch
        # loss for predicted action
        actions = actions.reshape(-1)
        states, next_states = self.gpu_loader.states_to_device((states, next_states))
        predicted_actions = self.predict_action(states, next_states)
        predicted_actions.reshape(-1, len(self.actions))
        action_loss = F.cross_entropy(predicted_actions, actions)

        # loss for predicted features
        all_states, _ = cat_states((states, next_states))
        features = self.features(all_states)
        current_features, next_features = th.chunk(features, 2, dim=0)
        predicted_features = self.predict_next_features(current_features, actions)
        feature_loss = F.mse_loss(predicted_features, next_features)

        loss = (1 - self.beta) * action_loss + self.beta * feature_loss
        metrics = {'Curiosity/loss_total': loss.detach().item(),
                   'Curiosity/loss_action_deduction': action_loss.detach().item(),
                   'Curiosity/loss_feature_prediction': feature_loss.detach().item()}
        return loss, metrics
