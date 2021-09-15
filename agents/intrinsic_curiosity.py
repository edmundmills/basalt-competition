from agents.soft_q import SoftQNetwork
from helpers.environment import ObservationSpace, ActionSpace
from torchvision.models.mobilenetv3 import mobilenet_v3_large

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

import wandb


class CuriosityModule(nn.Module):
    def __init__(self, n_observation_frames=1, eta=1):
        super().__init__()
        self.eta = eta
        self.actions = ActionSpace.actions()
        self.frame_shape = ObservationSpace.frame_shape
        self.n_observation_frames = n_observation_frames
        mobilenet_features = mobilenet_v3_large(pretrained=True, progress=True).features
        self.multi_frame_inuput = nn.Sequential(
            nn.Conv2d(3*self.n_observation_frames, 16, kernel_size=(3, 3),
                      stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.01,
                           affine=True, track_running_stats=True),
            nn.Hardswish()),
        self.single_frame_input = nn.Sequential(mobilenet_features[0])
        self.cnn = nn.Sequential(*mobilenet_features[1:])

        self.current_feature_dim = self._visual_features_dim(self.n_observation_frames)
        self.next_feature_dim = self._visual_features_dim(1)
        self.action_predictor = nn.Sequential(
            nn.Linear(sum(self.current_feature_dim, self.next_feature_dim), 256),
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
                dummy_input = self.single_frame_input(dummy_input)
            else:
                dummy_input = self.multi_frame_inuput(dummy_input)
            output = self.cnn(dummy_input)
        return output.size()

    def _visual_features_dim(self):
        return np.prod(list(self._visual_features_shape()))

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
            features = self.features(self.single_frame_input(pov))
        else:
            features = self.features(self.multi_frame_inuput(pov))
        features = features.flatten(start_dim=1)
        return features

    def reward(self, state, action, next_state, done):
        if done:
            return -1
        with th.no_grad():
            current_features = self.get_features(state)
            next_features = self.get_features(next_state, single_frame=True)
            predicted_next_features = self.predict_next_features(current_features, action)
            reward = self.eta * F.mse_loss(next_features, predicted_next_features)
        return reward


class IntrinsicCuriosityAgent:
    def __init__(self,
                 n_observation_frames=1,
                 discount_factor=0.99,
                 termination_critic=None):
        self.n_observation_frames = n_observation_frames
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.actor = SoftQNetwork(n_observation_frames=n_observation_frames,
                                  alpha=1).to(self.device)
        self.critic = SoftQNetwork(n_observation_frames=n_observation_frames,
                                   alpha=1).to(self.device)
        self.discount_factor = discount_factor
        self.curiosity_module = CuriosityModule(
            n_observation_frames=n_observation_frames).to(self.device)

    def get_action(self, state):
        state = [state_component.to(self.device) for state_component in state]
        with th.no_grad():
            Q = self.actor.get_Q(state)
            probabilities = self.actor.action_probabilities(Q).cpu().numpy().squeeze()
        action = np.random.choice(self.actions, p=probabilities)
        return action

    def random_action(self, obs, surpress_snowball=True):
        action = np.random.choice(self.actions)
        if surpress_snowball:
            while ActionSpace.threw_snowball(obs, action):
                action = np.random.choice(self.actions)
        return action

    def train(self, env, run, profiler=None):
        self.curiosity_optimizer = th.optim.Adam(self.curiosity_module.parameters(),
                                                 lr=run.config['curiosity_lr'])
        self.policy_optimizer = th.optim.Adam(self.actor.parameters(),
                                              lr=run.config['policy_lr'])
        self.q_optimizer = th.optim.Adam(self.critic.parameters(),
                                         lr=run.config['q_lr'])
        self.run = run
        replay_buffer = ReplayBuffer(reward=True,
                                     n_observation_frames=self.n_observation_frames)

        # th.autograd.set_detect_anomaly(True)
        obs = env.reset()
        replay_buffer.current_trajectory().append_obs(obs)
        current_state = replay_buffer.current_state()

        for step in range(self.run.config['training_steps']):
            iter_count = step + 1
            if iter_count <= self.run.config['start_steps']:
                action = self.random_action(
                    replay_buffer.current_trajectory().current_obs())
            else:
                action = self.get_action(current_state)

            replay_buffer.current_trajectory().actions.append(action)
            obs, _, done, _ = env.step(action)

            replay_buffer.current_trajectory().append_obs(obs)
            replay_buffer.current_trajectory().done = done

            next_state = replay_buffer.current_state()
            if step < self.run.config['start_steps']:
                reward = 0
            else:
                reward = self.curiosity_module.reward(current_state, next_state)
            if run.wandb:
                wandb.log({'reward': reward})

            replay_buffer.current_trajectory().rewards.append(reward)

            replay_buffer.increment_step()
            current_state = next_state

            batch_size = run.config['batch_size']
            if len(replay_buffer) >= batch_size:
                q_loss, policy_loss, curiosity_loss, average_Q = self.train_one_batch(
                    replay_buffer.sample(batch_size=batch_size))
                if run.wandb:
                    wandb.log({'policy_loss': policy_loss,
                               'q_loss': q_loss,
                               'curiosity_loss': curiosity_loss,
                               'average_Q': average_Q})

            if done:
                print(f'Trajectory completed at step {iter_count}')
                replay_buffer.new_trajectory()
                obs = env.reset()
                replay_buffer.current_trajectory().append_obs(obs)
                current_state = replay_buffer.current_state()

            if profiler:
                profiler.step()
            if (run.checkpoint_freqency and iter_count % run.checkpoint_freqency == 0
                    and iter_count < run.config['training_steps']):
                th.save(self.model.state_dict(), os.path.join('train', f'{run.name}.pth'))
                print(f'Checkpoint saved at step {iter_count}')

        print('Training complete')
        self.curiosity_optimizer = None
        self.policy_optimizer = None
        self.q_optimizer = None
        self.run = None

    def train_one_batch(self, batch):
        obs, actions, next_obs, done, reward = batch
        states = ObservationSpace.obs_to_state(obs)
        next_states = ObservationSpace.obs_to_state(next_obs)
        all_states = [th.cat(state_component, dim=0).to(self.device) for state_component
                      in zip(states, next_states)]
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.float().unsqueeze(1).to(self.device)

        # update critic
        all_Qs = self.critic.get_Q(all_states)
        Qs, next_Qs = th.chunk(all_Qs, 2, dim=0)
        next_Vs = self.critic.get_Vs(next_Qs)
        # use Qs only for taken actions
        Q_s_a = th.gather(Qs, dim=1, index=actions)
        target_Qs = rewards + self.discount_factor * next_Vs
        q_loss = F.mse_loss(Q_s_a, target_Qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # update actor
        entropies = self.actor.entropies(Qs)
        entropy_s_a = th.gather(entropies, dim=1, index=actions)
        policy_loss = -th.mean(Q_s_a - self.alpha * entropy_s_a)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        # update curiosity module
        # loss for predicted action
        states, next_states = th.chunk(all_states, 2, dim=0)
        predicted_actions = self.curiosity_module.predict_action(states, next_states)
        action_loss = F.cross_entropy(predicted_actions, actions)

        # loss for predicted features
        current_features = self.curiosity_module.get_features(states)
        next_features = self.curiosity_module.get_features(next_states, single_frame=True)
        predicted_features = self.curiosity_module.predicted_next_features(
            current_features, actions)
        feature_loss = F.mse_loss(predicted_features, next_features)

        curiosity_loss = action_loss + feature_loss
        self.curiosity_optimizer.zero_grad(set_to_none=True)
        curiosity_loss.backward()
        self.curiosity_optimizer.step()

        return q_loss.detach(), policy_loss.detach(), \
            curiosity_loss.detach(), all_qs.detach().mean().item()
