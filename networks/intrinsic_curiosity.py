from helpers.environment import ObservationSpace, ActionSpace
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from helpers.gpu import states_to_device

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F


class CuriosityModule(nn.Module):
    def __init__(self, n_observation_frames=1, eta=1):
        super().__init__()
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.eta = eta
        self.actions = ActionSpace.actions()
        self.frame_shape = ObservationSpace.frame_shape
        self.n_observation_frames = n_observation_frames
        mobilenet_features = mobilenet_v3_large(pretrained=True, progress=True).features
        self.multi_frame_features = nn.Sequential(
            nn.Sequential(nn.Conv2d(3*self.n_observation_frames, 16, kernel_size=(3, 3),
                                    stride=(2, 2), padding=(1, 1), bias=False),
                          nn.BatchNorm2d(16, eps=0.001, momentum=0.01,
                                         affine=True, track_running_stats=True),
                          nn.Hardswish()),
            *nn.Sequential(*mobilenet_features[1:])
        )
        self.single_frame_features = mobilenet_v3_large(
            pretrained=True, progress=True).features
        self.current_feature_dim = self._visual_features_dim(self.n_observation_frames)
        self.next_feature_dim = self._visual_features_dim(1)
        self.action_predictor = nn.Sequential(
            nn.Linear(self.current_feature_dim + self.next_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.actions))
        )
        self.feature_predictor = nn.Sequential(
            nn.Linear(self.current_feature_dim + len(self.actions), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.next_feature_dim)
        )

    def _visual_features_shape(self, frames):
        with th.no_grad():
            dummy_input = th.zeros((1, 3*frames, 64, 64))
            if frames == 1:
                dummy_features = self.single_frame_features(dummy_input)
            else:
                dummy_features = self.multi_frame_features(dummy_input)
        return dummy_features.size()

    def _visual_features_dim(self, *args):
        return np.prod(list(self._visual_features_shape(*args)))

    def predict_action(self, state, next_state):
        features = self.get_features(state)
        next_features = self.get_features(next_state, single_frame=True)
        x = th.cat((features, next_features), dim=1)
        action = self.action_predictor(x)
        return action

    def predict_next_features(self, features, action):
        one_hot_actions = F.one_hot(action, len(self.actions))
        x = th.cat((features, one_hot_actions), dim=1)
        features = self.feature_predictor(x)
        return features

    def get_features(self, state, single_frame=False):
        pov, _items = state
        if single_frame or self.n_observation_frames == 1:
            pov = pov[:, :3, :, :]
            features = self.single_frame_features(pov)
        else:
            features = self.multi_frame_features(pov)
        features = features.flatten(start_dim=1)
        return features

    def reward(self, state, action, next_state, done):
        if done:
            print('Done')
            return -1
        if ActionSpace.threw_snowball(state, action):
            print('Threw Snowball')
            return -1
        with th.no_grad():
            state = [state_component.to(self.device) for state_component in state]
            next_state = [state_component.to(self.device)
                          for state_component in next_state]
            action = th.tensor([action], device=self.device, dtype=th.int64).reshape(-1)
            current_features = self.get_features(state)
            next_features = self.get_features(next_state, single_frame=True)
            predicted_next_features = self.predict_next_features(current_features, action)
            reward = self.eta * F.mse_loss(next_features, predicted_next_features).item()
        return reward

    def loss(self, states, actions, next_states, _done, _rewards):
        # loss for predicted action
        states, next_states = states_to_device((states, next_states), self.device)
        predicted_actions = self.predict_action(states, next_states)
        action_loss = F.cross_entropy(predicted_actions, actions)

        # loss for predicted features
        current_features = self.get_features(states)
        next_features = self.get_features(next_states, single_frame=True)
        predicted_features = self.predict_next_features(current_features, actions)
        feature_loss = F.mse_loss(predicted_features, next_features)

        loss = action_loss + feature_loss
        return loss
