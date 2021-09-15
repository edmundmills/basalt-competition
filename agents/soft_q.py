from helpers.environment import ObservationSpace, ActionSpace
from agents.base_network import Network
from helpers.trajectories import Trajectory
from helpers.datasets import MixedReplayBuffer

import wandb
import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np

import math
import os


class SoftQNetwork(Network):
    def __init__(self, alpha, n_observation_frames):
        super().__init__(n_observation_frames)
        self.alpha = alpha

    def get_Q(self, state):
        return self.forward(state)

    def get_V(self, Qs):
        v = self.alpha * th.logsumexp(Qs / self.alpha, dim=1, keepdim=True)
        return v

    def action_probabilities(self, Qs):
        probabilities = F.softmax(Qs/self.alpha, dim=1)
        return probabilities

    def entropies(self, Qs):
        entropies = F.log_softmax(Qs/self.alpha, dim=1)
        return entropies


class SoftQAgent:
    def __init__(self, alpha=1, termination_critic=None,
                 discount_factor=0.99, n_observation_frames=1):
        self.actions = ActionSpace.actions()
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.model = SoftQNetwork(
            alpha=self.alpha, n_observation_frames=n_observation_frames).to(self.device)
        self.termination_critic = termination_critic
        self.n_observation_frames = n_observation_frames

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

    def train(self, env, run, expert_dataset, profiler=None):
        self.optimizer = th.optim.Adam(self.model.parameters(),
                                       lr=run.config['learning_rate'])
        self.run = run
        replay_buffer = MixedReplayBuffer(expert_dataset=expert_dataset,
                                          batch_size=run.config['batch_size'],
                                          expert_sample_fraction=0.5,
                                          n_observation_frames=self.n_observation_frames)

        # th.autograd.set_detect_anomaly(True)
        obs = env.reset()
        replay_buffer.current_trajectory().append_obs(obs)

        for step in range(self.run.config['training_steps']):
            iter_count = step + 1

            current_obs = replay_buffer.current_trajectory().current_obs(
                n_observation_frames=self.n_observation_frames)
            current_state = ObservationSpace.obs_to_state(current_obs)
            action = self.get_action(current_state)
            if ActionSpace.threw_snowball(current_obs, action):
                print(f'Threw Snowball at step {iter_count}')
                if self.termination_critic is not None:
                    reward = self.termination_critic.termination_reward(current_state)
                    print(f'Termination reward: {reward:.2f}')
                    if run.wandb:
                        wandb.log({'termination_reward': reward})
                else:
                    reward = 0
            else:
                reward = 0

            replay_buffer.current_trajectory().actions.append(action)
            replay_buffer.current_trajectory().rewards.append(reward)
            obs, _, done, _ = env.step(action)
            replay_buffer.current_trajectory().append_obs(obs)
            replay_buffer.current_trajectory().done = done
            replay_buffer.increment_step()
            if len(replay_buffer) >= replay_buffer.replay_batch_size:
                loss = self.train_one_batch(replay_buffer.sample_expert(),
                                            replay_buffer.sample_replay())
                if run.wandb:
                    wandb.log({'loss': loss})
                run.append_loss(loss.detach().item())

            run.print_update(iter_count)

            if done:
                print(f'Trajectory completed at step {iter_count}')
                replay_buffer.new_trajectory()
                obs = env.reset()
                replay_buffer.current_trajectory().append_obs(obs)

            if profiler:
                profiler.step()
            if (run.checkpoint_freqency and iter_count % run.checkpoint_freqency == 0
                    and iter_count < run.config['training_steps']):
                th.save(self.model.state_dict(), os.path.join('train', f'{run.name}.pth'))
                print(f'Checkpoint saved at step {iter_count}')

        print('Training complete')
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

        mask = (expert_actions != -1).squeeze()
        expert_actions = expert_actions[mask]

        expert_batch_size = expert_actions.size()[0]
        replay_batch_size = replay_actions.size()[0]

        batch_actions = th.cat([expert_actions,
                                replay_actions], dim=0).to(self.device)

        expert_states = ObservationSpace.obs_to_state(expert_obs)
        expert_next_states = ObservationSpace.obs_to_state(expert_next_obs)
        replay_states = ObservationSpace.obs_to_state(replay_obs)
        replay_next_states = ObservationSpace.obs_to_state(replay_next_obs)
        expert_states = [state_component[mask] for state_component in expert_states]
        expert_next_states = [state_component[mask]
                              for state_component in expert_next_states]

        batch_states = [th.cat(state_component, dim=0).to(self.device) for state_component
                        in zip(expert_states, replay_states,
                               expert_next_states, replay_next_states)]

        expert_rewards = th.ones(expert_batch_size, 1)
        replay_rewards = replay_rewards.float().unsqueeze(1)
        batch_rewards = th.cat([expert_rewards,
                                replay_rewards], dim=0).to(self.device)

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

        mask = (expert_actions != -1).squeeze()
        expert_actions = expert_actions[mask].to(self.device)

        expert_batch_size = expert_actions.size()[0]
        replay_batch_size = replay_actions.size()[0]

        expert_states = ObservationSpace.obs_to_state(expert_obs)
        expert_next_states = ObservationSpace.obs_to_state(expert_next_obs)
        replay_states = ObservationSpace.obs_to_state(replay_obs)
        replay_next_states = ObservationSpace.obs_to_state(replay_next_obs)
        expert_states = [state_component[mask] for state_component in expert_states]
        expert_next_states = [state_component[mask]
                              for state_component in expert_next_states]

        batch_states = [th.cat(state_component, dim=0).to(self.device) for state_component
                        in zip(expert_states, replay_states,
                               expert_next_states, replay_next_states)]

        batch_lengths = [expert_batch_size, replay_batch_size,
                         expert_batch_size, replay_batch_size]

        batch_Qs = self.model.get_Q(batch_states)

        Q_expert, _, _, _ = th.split(batch_Qs, batch_lengths, dim=0)
        predicted_Q_expert = th.gather(Q_expert, 1, expert_actions)

        batch_Vs = self.model.get_V(batch_Qs)
        V_expert, V_replay, V_next_expert, V_next_replay = th.split(
            batch_Vs, batch_lengths, dim=0)

        def distance_function(x):
            return x - 1/4 * x**2

        objective = -(th.mean(distance_function(predicted_Q_expert -
                                                self.discount_factor * V_next_expert)) -
                      th.mean(th.cat((V_replay, V_expert), dim=0) -
                              self.discount_factor * th.cat((V_next_replay,
                                                             V_next_expert), dim=0)))

        # # Add an additional term to the loss for the reward of the throw snoball actions
        # replay_rewards = replay_rewards.unsqueeze(1).float().to(self.device)
        # rewards_mask = replay_rewards != 0.
        # replay_rewards = replay_rewards[rewards_mask]
        # if replay_rewards.size()[0] > 0:
        #     predicted_Q_replay = th.gather(Q_replay, 1, replay_actions)
        #     predicted_r = (predicted_Q_replay[rewards_mask]
        #                    - self.discount_factor * V_next_replay[rewards_mask])
        #     objective += F.mse_loss(predicted_r, replay_rewards)

        self.optimizer.zero_grad(set_to_none=True)
        objective.backward()
        self.optimizer.step()
        return objective.detach()
