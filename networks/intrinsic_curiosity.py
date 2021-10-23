from core.environment import ObservationSpace, ActionSpace
from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from core.gpu import GPULoader, cat_states

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F


class CuriosityModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.n_observation_frames = config.n_observation_frames
        # scales the returned reward
        self.eta = 0.5
        # controls the relative weight of the two different loss loss_functions
        # low values prioritize deducing actions over predicting next features
        self.beta = 0.1
        self.actions = ActionSpace.actions()
        self.frame_shape = ObservationSpace.frame_shape
        mobilenet_features = mobilenet_v3_small(pretrained=True, progress=True).features
        self.features = nn.Sequential(
            nn.Sequential(nn.Conv2d(3*self.n_observation_frames, 16, kernel_size=(3, 3),
                                    stride=(2, 2), padding=(1, 1), bias=False),
                          nn.BatchNorm2d(16, eps=0.001, momentum=0.01,
                                         affine=True, track_running_stats=True),
                          nn.Hardswish()),
            *nn.Sequential(*mobilenet_features[1:4]),
            nn.AvgPool2d(2)
        )
        self.feature_dim = self._visual_features_dim(self.n_observation_frames)
        self.action_predictor = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.actions))
        )
        self.feature_predictor = nn.Sequential(
            nn.Linear(self.feature_dim + len(self.actions), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim)
        )
        self.gpu_loader = GPULoader(config)

    def _visual_features_shape(self, frames):
        with th.no_grad():
            dummy_input = th.zeros((1, 3*frames, 64, 64))
            dummy_features = self.features(dummy_input)
        print('Curiosity Module Feature Shape: ', dummy_features.size())
        return dummy_features.size()

    def _visual_features_dim(self, *args):
        return np.prod(list(self._visual_features_shape(*args)))

    def predict_action(self, state, next_state):
        states, _ = cat_states((state, next_state))
        features = self.get_features(states)
        current_features, next_features = th.chunk(features, 2, dim=0)
        x = th.cat((current_features, next_features), dim=1)
        action = self.action_predictor(x)
        return action

    def predict_next_features(self, features, action):
        one_hot_actions = F.one_hot(action, len(self.actions))
        x = th.cat((features, one_hot_actions), dim=1)
        features = self.feature_predictor(x)
        return features

    def get_features(self, state):
        pov = state[0]
        features = self.features(pov).flatten(start_dim=1)
        return features

    def reward(self, state, action, next_state, done):
        if done:
            return 0
        if ActionSpace.threw_snowball(state, action):
            return 0
        states = [state_component.unsqueeze(0) for state_component in state]
        next_states = [state_component.unsqueeze(0) for state_component in next_state]
        states = self.gpu_loader.states_to_device((states, next_states))
        action = th.tensor([action], device=self.device, dtype=th.int64).reshape(-1)
        states, _ = cat_states(states)
        with th.no_grad():
            features = self.get_features(states)
            current_features, next_features = th.chunk(features, 2, dim=0)
            predicted_next_features = self.predict_next_features(current_features, action)
            reward = self.eta * F.mse_loss(next_features,
                                           predicted_next_features).item()
        reward = min(max(reward, -1), 1)
        return reward

    def bulk_rewards(self, batch, expert=False):
        state, actions, next_state, done, reward = batch
        actions = actions.reshape(-1)
        states, _ = cat_states((state, next_state))
        with th.no_grad():
            features = self.get_features(states)
            current_features, next_features = th.chunk(features, 2, dim=0)
            predicted_next_features = self.predict_next_features(current_features,
                                                                 actions)
            reward = F.mse_loss(next_features, predicted_next_features, reduction='none')
            reward = th.mean(reward, dim=1, keepdim=True)
        threw_snowball = ActionSpace.threw_snowball_tensor(state, actions, self.device)
        if expert:
            reward = reward * (1 - threw_snowball) + threw_snowball
        else:
            reward = reward * (1 - threw_snowball) - threw_snowball
        return reward.squeeze().tolist()

    def loss(self, batch):
        states, actions, next_states, _done, _rewards = batch
        # loss for predicted action
        actions = actions.reshape(-1)
        states, next_states = self.gpu_loader.states_to_device((states, next_states))
        predicted_actions = self.predict_action(states, next_states)
        action_loss = F.cross_entropy(predicted_actions, actions)

        # loss for predicted features
        all_states, _ = cat_states((states, next_states))
        features = self.get_features(all_states)
        current_features, next_features = th.chunk(features, 2, dim=0)
        predicted_features = self.predict_next_features(current_features, actions)
        feature_loss = F.mse_loss(predicted_features, next_features)

        loss = (1 - self.beta) * action_loss + self.beta * feature_loss
        metrics = {'curiosity_loss_total': loss.detach().item(),
                   'curiosity_loss_action_deduction': action_loss.detach().item(),
                   'curiosity_loss_feature_prediction': feature_loss.detach().item()}
        return loss, metrics
