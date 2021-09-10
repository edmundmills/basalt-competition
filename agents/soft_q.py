from helpers.environment import ObservationSpace, ActionSpace
from agents.base_network import Network
from helpers.trajectories import Trajectory
from helpers.datasets import MixedReplayBuffer

import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np

import math
import os


class SoftQNetwork(Network):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

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
    def __init__(self, alpha=1, termination_critic=None, discount_factor=0.99):
        self.actions = ActionSpace.actions()
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.model = SoftQNetwork(alpha=self.alpha).to(self.device)
        self.termination_critic = termination_critic

    def load_parameters(self, model_file_path):
        self.model.load_state_dict(
            th.load(model_file_path, map_location=self.device), strict=False)

    def save(self, path):
        th.save(self.model.state_dict(), path)

    def get_action(self, state):
        state = [state_component.to(self.device) for state_component in state]
        with th.no_grad():
            Q = self.model.get_Q(state)
            probabilities = self.model.action_probabilities(Q).cpu().numpy().squeeze()
        action = np.random.choice(self.actions, p=probabilities)
        return action

    def train(self, env, run, profiler=None):
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=run.lr)
        self.run = run
        replay_buffer = MixedReplayBuffer(capacity=1e6, batch_size=64,
                                          expert_sample_fraction=0.5)

        obs = env.reset()
        current_trajectory = Trajectory()
        current_trajectory.obs.append(obs)

        for step in range(self.run.training_steps):
            iter_count = step + 1

            current_obs = current_trajectory.current_obs()
            current_state = ObservationSpace.obs_to_state(current_obs)
            action = self.get_action(current_state)
            if action == 11:
                print(f'Threw Snowball at step {iter_count}')
                if self.termination_critic is not None:
                    reward = self.termination_critic.termination_reward(current_state)
                    print(f'Termination reward: {reward:.2f}')
                else:
                    reward = 0
            else:
                reward = 0

            current_trajectory.actions.append(action)
            obs, _, done, _ = env.step(action)
            current_trajectory.obs.append(obs)
            current_trajectory.done = done
            next_obs = current_trajectory.current_obs()
            replay_buffer.push(current_obs, action, next_obs, done, reward)

            if len(replay_buffer) >= replay_buffer.replay_batch_size:
                loss = self.train_one_batch(replay_buffer.sample_expert(),
                                            replay_buffer.sample_replay())
                run.append_loss(loss.detach().item())

            run.print_update(iter_count)

            if done:
                print(f'Trajectory completed at step {iter_count}')
                obs = env.reset()
                currnet_trajectory = Trajectory()
                current_trajectory.obs.append(obs)

            if profiler:
                profiler.step()
            if (run.checkpoint_freqency and iter_count % run.checkpoint_freqency == 0
                    and iter_count < run.training_steps):
                th.save(self.model.state_dict(), os.path.join('train', f'{run.name}.pth'))
                print(f'Checkpoint saved at step {iter_count}')

        print('Training complete')
        th.save(self.model.state_dict(), os.path.join('train', f'{run.name}.pth'))
        self.run.save_data()
        self.optimizer = None
        self.run = None


class SqilAgent(SoftQAgent):
    def train_one_batch(self, expert_batch, replay_batch):
        expert_obs, expert_actions, expert_next_obs, _expert_done = expert_batch
        (replay_obs, replay_actions, replay_next_obs,
         _replay_done, replay_rewards) = replay_batch

        expert_actions = ActionSpace.dataset_action_batch_to_actions(expert_actions)
        expert_actions = th.from_numpy(expert_actions).unsqueeze(1)
        replay_actions = replay_actions.unsqueeze(1)

        expert_states = ObservationSpace.obs_to_state(expert_obs)
        expert_next_states = ObservationSpace.obs_to_state(expert_next_obs)
        replay_states = ObservationSpace.obs_to_state(replay_obs)
        replay_next_states = ObservationSpace.obs_to_state(replay_next_obs)

        # remove expert no-op actions
        mask = (expert_actions != -1).squeeze()
        expert_actions = expert_actions[mask]
        expert_states = [state_component[mask] for state_component in expert_states]
        expert_next_states = [state_component[mask]
                              for state_component in expert_next_states]

        masked_expert_batch_size = len(expert_actions)
        expert_rewards = th.ones(masked_expert_batch_size, 1, device=self.device)
        replay_rewards = replay_rewards.unsqueeze(1).float().to(self.device)

        batch_rewards = th.cat([expert_rewards,
                                replay_rewards], dim=0)
        batch_actions = th.cat([expert_actions.to(self.device),
                                replay_actions.to(self.device)], dim=0)
        batch_states = [th.cat(state_component, dim=0).to(self.device) for state_component
                        in zip(expert_states, replay_states,
                               expert_next_states, replay_next_states)]

        batch_Qs = self.model.get_Q(batch_states)
        current_Qs, next_Qs = th.chunk(batch_Qs, 2, dim=0)
        predicted_Qs = th.gather(current_Qs, 1, batch_actions)

        V_next = self.model.get_V(next_Qs)
        y = batch_rewards + self.discount_factor * V_next

        objective = F.mse_loss(predicted_Qs, y)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        return objective.detach()


class IQLearnAgent(SoftQAgent):
    def train_one_batch(self, expert_batch, replay_batch):
        expert_obs, expert_actions, expert_next_obs, _expert_done = expert_batch
        (replay_obs, replay_actions, replay_next_obs,
         _replay_done, replay_rewards) = replay_batch

        expert_actions = ActionSpace.dataset_action_batch_to_actions(expert_actions)
        expert_actions = th.from_numpy(expert_actions).unsqueeze(1)
        replay_actions = replay_actions.unsqueeze(1)

        expert_states = ObservationSpace.obs_to_state(expert_obs)
        expert_next_states = ObservationSpace.obs_to_state(expert_next_obs)
        replay_states = ObservationSpace.obs_to_state(replay_obs)
        replay_next_states = ObservationSpace.obs_to_state(replay_next_obs)

        # remove expert no-op actions
        mask = (expert_actions != -1).squeeze()
        expert_actions = expert_actions[mask]
        expert_states = [state_component[mask] for state_component in expert_states]
        expert_next_states = [state_component[mask]
                              for state_component in expert_next_states]

        masked_expert_batch_size = len(expert_actions)
        replay_batch_size = len(replay_actions)

        batch_states = [th.cat(state_component, dim=0).to(self.device) for state_component
                        in zip(expert_states, replay_states,
                               expert_next_states, replay_next_states)]
        batch_lengths = [masked_expert_batch_size, replay_batch_size,
                         masked_expert_batch_size, replay_batch_size]

        batch_Qs = self.model.get_Q(batch_states)

        Q_expert, Q_replay, _, _ = th.split(batch_Qs, batch_lengths, dim=0)
        predicted_Q_expert = th.gather(Q_expert, 1, expert_actions.to(self.device))
        predicted_Q_replay = th.gather(Q_replay, 1, replay_actions.to(self.device))

        batch_Vs = self.model.get_V(batch_Qs)
        _, V_replay, V_next_expert, V_next_replay = th.split(
            batch_Vs, batch_lengths, dim=0)

        def distance_function(x):
            return x - 1/4 * x**2

        objective = -(th.mean(distance_function(predicted_Q_expert -
                                                self.discount_factor * V_next_expert)
                              ) -
                      th.mean(V_replay - self.discount_factor * V_next_replay))

        # Add an additional term to the loss for the reward of the throw snoball actions
        replay_rewards = replay_rewards.unsqueeze(1).float().to(self.device)
        rewards_mask = replay_rewards != 0.
        replay_rewards = replay_rewards[rewards_mask]
        if replay_rewards.size()[0] > 0:
            predicted_r = (predicted_Q_replay[rewards_mask]
                           - self.discount_factor * V_next_replay[rewards_mask])
            objective += F.mse_loss(predicted_r, replay_rewards)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        return objective.detach()
