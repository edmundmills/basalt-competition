from agents.soft_q import SoftQNetwork
from helpers.environment import ObservationSpace, ActionSpace
from torchvision.models.mobilenetv3 import mobilenet_v3_large

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

import wandb


class CuriosityModule(nn.Module):
    def __init__(self, eta=1):
        super().__init__()
        self.n_observation_frames = 1
        self.eta = eta
        self.actions = ActionSpace.actions()
        self.frame_shape = ObservationSpace.frame_shape
        self.item_dim = 2 * len(ObservationSpace.items())
        self.output_dim = len(self.actions)
        self.cnn = mobilenet_v3_large(pretrained=True, progress=True).features
        self.feature_dim = self._visual_features_dim()
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

    def _visual_features_shape(self):
        with th.no_grad():
            dummy_input = th.zeros((1, 3, 64, 64))
            output = self.cnn(dummy_input)
        return output.size()

    def _visual_features_dim(self):
        return np.prod(list(self._visual_features_shape()))

    def predict_action(self, state, next_state):
        both_states = [th.cat(state_component, dim=0)
                       for state_component in zip(state, next_state)]
        all_features = self.get_features(both_states)
        features, next_features = th.chunk(all_features, 2, dim=0)
        x = th.cat(features, next_features, dim=1)
        action = self.action_predictor(x)
        return action

    def predict_next_features(self, features, action):
        x = th.cat((features, action), dim=1)
        features = self.feature_predictor(x).flatten(start_dim=1)
        return features

    def get_features(self, state):
        pov, _items = state
        current_pov = pov[:, :3, :, :]
        features = self.features(current_pov).flatten(start_dim=1)
        return features

    def reward(self, state, action, next_state, done):
        if done:
            return -1
        with th.no_grad():
            current_features = self.get_features(state)
            next_features = self.get_features(next_state)
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
        self.curiosity_module = CuriosityModule().to(self.device)

    def get_action(self, state):
        state = [state_component.to(self.device) for state_component in state]
        with th.no_grad():
            Q = self.actor.get_Q(state)
            probabilities = self.actor.action_probabilities(Q).cpu().numpy().squeeze()
        action = np.random.choice(self.actions, p=probabilities)
        return action

    def random_action(self, obs, surpress_termination=True):
        action = np.random.choice(self.actions)
        if surpress_termination:
            while ActionSpace.threw_snowball(obs, action):
                action = np.random.choice(self.actions)
        return action

    def train(self, env, run, profiler=None):
        self.curiosity_optimizer = th.optim.Adam(self.curiosity_module.parameters(),
                                                 lr=run.config['curiosity_lr'])
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(),
                                             lr=run.config['policy_lr'])
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(),
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
            replay_buffer.current_trajectory().rewards.append(reward)

            replay_buffer.increment_step()
            current_state = next_state

            batch_size = run.config['batch_size']
            if len(replay_buffer) >= batch_size:
                loss = self.train_one_batch(replay_buffer.sample(batch_size=batch_size))
                if run.wandb:
                    wandb.log({'loss': loss})
                run.append_loss(loss.detach().item())

            run.print_update(iter_count)

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
        self.optimizer = None
        self.run = None

    def train_one_batch(self, batch):
        state, actions, next_state, done, reward = batch
        states = ObservationSpace.obs_to_state(expert_obs)
        next_states = ObservationSpace.obs_to_state(expert_next_obs)
        batch_states = [th.cat(state_component, dim=0).to(self.device) for state_component
                        in zip(states, next_states)]
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.float().unsqueeze(1).to(self.device)
        states, next_states = th.chunk(batch_states, 2, dim=0)
        one_hot_actions = F.one_hot(actions, len(self.actions))

        # update critic

        critic_loss =
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor

        actor_loss =
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # update curiosity
        predicted_actions = self.curiosity_module.predict_action(states, next_states)
        action_loss = F.cross_entropy(predicted_actions, actions)

        batch_features = self.curiosity_module.get_features(batch_states)
        current_features, next_features = th.chunk(batch_features, 2, dim=0)
        predicted_features = self.curiosity_module.predicted_next_features(
            current_features, one_hot_actions)
        feature_loss = F.mse_loss(predicted_features, next_features)

        curiosity_loss = action_loss + feature_loss
        self.curiosity_optimizer.zero_grad(set_to_none=True)
        curiosity_loss.backward()
        self.curiosity_optimizer.step()
