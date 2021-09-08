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
        th.save(self.model.state_dict(), path)

    def get_action(self, state):
        state = [state_component.to(self.device) for state_component in state]
        with th.no_grad():
            Q = self.model.get_Q(state)
            probabilities = self.model.action_probabilities(Q).cpu().numpy().squeeze()
        action = np.random.choice(self.actions, p=probabilities)
        print(action)
        return action

    def train(self, env, run):
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=run.lr)
        self.run = run
        replay_buffer = MixedReplayBuffer(capacity=1e6, batch_size=64,
                                          expert_sample_fraction=0.5)

        obs = env.reset()
        current_trajectory = Trajectory()
        current_trajectory.obs.append(obs)

        for step in range(self.run.training_steps):
            current_obs = current_trajectory.current_obs()
            current_state = ObservationSpace.obs_to_state(current_obs)
            action = self.get_action(current_state)
            if action == 11:
                print(f'Threw Snowball at step {step}')
                if self.termination_critic is not None:
                    reward = self.termination_critic.termination_reward(current_state)
                    print(f'Termination reward: {reward:.2f}')

            current_trajectory.actions.append(action)
            obs, _, done, _ = env.step(action)
            current_trajectory.obs.append(obs)
            current_trajectory.done = done
            next_obs = current_trajectory.current_obs()
            replay_buffer.push(current_obs, action, next_obs, done)

            if len(replay_buffer) >= replay_buffer.replay_batch_size:
                loss = self.train_one_batch(replay_buffer.sample_expert(),
                                            replay_buffer.sample_replay())
                run.append_loss(loss.detach().item())

            run.print_update(step)

            if done:
                print(f'Trajectory completed at step {step}')
                obs = env.reset()
                currnet_trajectory = Trajectory()
                current_trajectory.obs.append(obs)

        print('Training complete')
        th.save(self.model.state_dict(), os.path.join('train', f'{run.name}.pth'))
        self.run.save_data()
        self.optimizer = None
        self.run = None


class SqilAgent(SoftQAgent):
    def train_one_batch(self, expert_batch, replay_batch):
        expert_obs, expert_actions, expert_next_obs, _expert_done = expert_batch
        replay_obs, replay_actions, replay_next_obs, _replay_done = replay_batch

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

        expert_rewards = th.ones(masked_expert_batch_size, 1)
        replay_rewards = th.zeros(replay_batch_size, 1)

        # update replay rewards with termination critic
        if self.termination_critic is not None:
            threw_snowball = ActionSpace.threw_snowball_list(replay_obs, replay_actions)
            for idx, termination in enumerate(threw_snowball):
                if termination:
                    termination_state = [state_component[idx].unsqueeze(0)
                                         for state_component in replay_states]
                    reward = self.termination_critic.termination_reward(
                        termination_state)
                    replay_rewards[idx] = reward

        batch_rewards = th.cat([expert_rewards.to(self.device),
                                replay_rewards.to(self.device)], dim=0)
        batch_actions = th.cat([expert_actions.to(self.device),
                                replay_actions.to(self.device)], dim=0)
        batch_states = [th.cat(state_component, dim=0).to(self.device) for state_component
                        in zip(expert_states, replay_states,
                               expert_next_states, replay_next_states)]

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
