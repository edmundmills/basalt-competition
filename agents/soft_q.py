from helpers.environment import ObservationSpace, ActionSpace
from helpers.trajectory import Trajectory
from helpers.datasets import MixedReplayBuffer
from torchvision.models.mobilenetv3 import mobilenet_v3_large

import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np

import math


class SoftQNetwork(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.frame_shape = ObservationSpace.frame_shape
        self.inventory_dim = len(ObservationSpace.items())
        self.equip_dim = len(ObservationSpace.items())
        self.output_dim = len(ActionSpace.actions())
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

    def get_Q(self, state):
        return self.forward(state)

    def get_V(self, Qs):
        v = self.alpha * \
            th.log(th.sum(th.exp(Qs / self.alpha), dim=1, keepdim=True))
        return v

    def action_probabilities(self, Qs):
        Vs = self.get_V(Qs).repeat(1, len(self.actions))
        probabilities = th.exp((Qs - Vs)/self.alpha)
        probabilities /= th.sum(probabilities)
        return probabilities


class SoftQAgent:
    def __init__(self, alpha=1, termination_critic=None):
        self.actions = ActionSpace.actions()
        self.alpha = alpha
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.model = SoftQNetwork(alpha=self.alpha).to(self.device)
        self.termination_critic = termination_critic

    def load_parameters(self, model_file_path):
        self.model.load_state_dict(
            th.load(model_file_path, map_location=self.device), strict=False)

    def save(self, path):
        th.save(agent.model.state_dict(), path)

    def get_action(self, obs):
        states = ObservationSpace.obs_to_states(obs)
        with th.no_grad():
            Q = self.model.get_Q(states)
            probabilities = self.model.action_probabilities(Q).cpu().numpy().squeeze()
        action = np.random.choice(self.actions, p=probabilities)
        return action

    def train(self, env, expert_data_path, run):
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=run.lr)
        self.run = run

        replay_buffer = MixedReplayBuffer(capacity=10000000, batch_size=64,
                                          expert_data_path=expert_data_path,
                                          expert_sample_fraction=0.5)

        obs = env.reset()
        current_trajectory = Trajectory()
        current_trajectory.obs.append(obs)

        for step in range(self.run.training_steps):
            current_obs = trajectory.current_obs()
            action = self.get_action(current_obs)
            if action == 11:
                print(f'Threw Snowball at step {step}')
            current_trajectory.actions.append(action)
            obs, _, done, _ = env.step(action)
            current_trajectory.obs.append(obs)
            current_trajectory.done = done
            next_obs = trajectory.current_obs()
            replay_buffer.push(current_obs, action, next_obs, done)

            if len(replay_buffer) >= replay_buffer.replay_batch_size:
                loss = agent.train_one_batch(replay_buffer.sample_expert(),
                                             replay_buffer.sample_replay())
                run.append_loss(loss.detach().item())

            run.print_update()

            if done:
                print(f'Trajectory completed at step {step}')
                obs = env.reset()
                currnet_trajectory = Trajectory()
                current_trajectory.obs.append(obs)

        self.save()
        self.run.save_data()
        self.optimizer = None
        self.run = None


class SqilAgent(SoftQAgent):
    def train_one_batch(self, expert_batch, replay_batch):
        expert_obs, expert_actions, expert_next_obs, _expert_done = expert_batch
        replay_obs, replay_actions, replay_next_obs, _replay_done = replay_batch

        expert_actions = ActionSpace.dataset_action_batch_to_actions(expert_actions)
        expert_actions = th.from_numpy(expert_actions).unsqueeze(1).to(self.device)

        # remove expert no-op actions
        mask = expert_actions != -1
        expert_obs = expert_obs[mask]
        expert_actions = expert_actions[mask]
        expert_next_obs = expert_next_obs[mask]
        masked_expert_batch_size = len(expert_actions)
        replay_batch_size = len(replay_actions)

        expert_states = ObservationSpace.obs_to_states(expert_obs).to(self.device)
        expert_next_states = ObservationSpace.obs_to_states(
            expert_next_obs).to(self.device)
        replay_states = ObservationSpace.obs_to_states(replay_obs).to(self.device)
        replay_next_states = ObservationSpace.obs_to_states(
            replay_next_obs).to(self.device)

        expert_rewards = th.ones(masked_expert_batch_size, 1)
        replay_rewards = th.zeros(replay_batch_size, 1)

        # update replay rewards with termination critic
        if self.termination_critic is not None:
            threw_snowball = ActionSpace.threw_snowball(replay_obs, replay_actions)
            for idx, termination in enumerate(threw_snowball):
                if termination:
                    reward = self.termination_critic.termination_reward(
                        replay_states[idx])
                    replay_rewards[idx] = reward

        batch_rewards = th.cat([expert_rewards, replay_rewards], dim=0).to(self.device)
        batch_actions = th.cat([expert_actions, replay_actions], dim=0)
        batch_states = th.cat([expert_states, replay_states,
                               expert_next_states, replay_next_states], dim=0)

        batch_Qs = self.model.get_Q(batch_states)
        current_Qs, next_Qs = th.chunk(batch_Qs, 2, dim=0)

        predicted_Qs = th.gather(current_Qs, 1, batch_actions)
        V_next = self.model.get_V(next_Qs)
        y = batch_rewards + self.run.discount_factor * V_next

        objective = F.mse_loss(predicted_Qs, y)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        return objective.detach()


class IQLearnAgent(SoftQAgent):
    def train_one_batch(self, expert_batch, replay_batch):
        expert_states, expert_actions, expert_next_states, _expert_done = expert_batch
        replay_states, replay_actions, replay_next_states, _replay_done = replay_batch
        replay_states, replay_actions, replay_next_states = (
            replay_states.to(self.device), replay_actions.to(self.device),
            replay_next_states.to(self.device))
        expert_actions = dataset_action_batch_to_actions(expert_actions)
        expert_states = dataset_obs_batch_to_obs(expert_states).to(self.device)
        expert_next_states = dataset_obs_batch_to_obs(expert_next_states).to(self.device)

        # remove expert no-op actions
        mask = expert_actions != -1
        expert_states = expert_states[mask]
        expert_actions = expert_actions[mask]
        expert_next_states = expert_next_states[mask]
        masked_expert_batch_size = len(expert_actions)
        replay_batch_size = len(replay_actions)

        all_states = th.cat([expert_states, expert_next_states,
                            replay_states, replay_next_states], dim=0)
        all_Qs = self.model.get_Q(all_states, grad=True)
        expert_Qs, expert_Qs_next, replay_Qs, replay_Qs_next = th.split(
            all_Qs, [masked_expert_batch_size, masked_expert_batch_size,
                     replay_batch_size, replay_batch_size], dim=0)

        Q_expert_s_a = expert_Qs[th.arange(len(expert_actions)),
                                 expert_actions].unsqueeze(1)
        V_next_expert = self.model.get_V(expert_Qs_next)
        V_replay = self.model.get_V(replay_Qs)
        V_replay_next = self.model.get_V(replay_Qs_next)

        def distance_function(x):
            return x - 1/4 * x**2

        objective = -(th.mean(distance_function(Q_expert_s_a -
                                                self.run.discount_factor * V_next_expert)
                              ) -
                      th.mean(V_replay - self.run.discount_factor * V_replay_next))

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        return objective.detach()
