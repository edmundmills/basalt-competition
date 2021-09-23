from networks.soft_q import SoftQNetwork, TwinnedSoftQNetwork
from algorithms.algorithm import Algorithm
from algorithms.loss_functions.sac import SACQLoss, SACPolicyLoss
from algorithms.loss_functions.iqlearn import IQLearnLossSAC
from networks.intrinsic_curiosity import CuriosityModule
from helpers.environment import ObservationSpace, ActionSpace
from helpers.datasets import ReplayBuffer, MixedReplayBuffer
from helpers.gpu import states_to_device, disable_gradients

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
import os
from pathlib import Path


class SoftActorCritic(Algorithm):
    def __init__(self, config, actor=None,
                 initial_replay_buffer=None, initial_iter_count=0):
        super().__init__(config)
        self.starting_steps = config.starting_steps
        self.suppress_snowball_steps = config.suppress_snowball_steps
        self.training_steps = config.training_steps
        self.batch_size = config.batch_size
        self.tau = config.tau
        self.double_q = config.double_q
        self.target_update_interval = 1
        self.updates_per_step = 1
        self.curiosity_pretraining_steps = 0
        # Set up replay buffer
        if initial_replay_buffer is None:
            self.replay_buffer = ReplayBuffer(
                reward=True, n_observation_frames=config.n_observation_frames)
        else:
            self.replay_buffer = initial_replay_buffer
        self.iter_count += initial_iter_count

        # Set up networks - actor
        if actor is not None:
            self.actor = actor.to(self.device)
        else:
            self.actor = SoftQNetwork(
                n_observation_frames=config.n_observation_frames,
                alpha=config.alpha).to(self.device)

        # Set up networks - critic
        if self.double_q:
            self.online_q = TwinnedSoftQNetwork(
                n_observation_frames=config.n_observation_frames,
                alpha=config.alpha).to(self.device)
            self.target_q = TwinnedSoftQNetwork(
                n_observation_frames=config.n_observation_frames,
                alpha=config.alpha).to(self.device)
        else:
            self.online_q = SoftQNetwork(
                n_observation_frames=config.n_observation_frames,
                alpha=config.alpha).to(self.device)
            self.target_q = SoftQNetwork(
                n_observation_frames=config.n_observation_frames,
                alpha=config.alpha).to(self.device)

        self.target_q.load_state_dict(self.online_q.state_dict())
        disable_gradients(self.target_q)

        # Loss functions
        self._q_loss = SACQLoss(self.online_q, self.target_q, config)
        self._policy_loss = SACPolicyLoss(self.actor, self.online_q, config)

        # Optimizers
        self.policy_optimizer = th.optim.Adam(self.actor.parameters(),
                                              lr=config.policy_lr)
        self.q_optimizer = th.optim.Adam(self.online_q.parameters(),
                                         lr=config.q_lr)

    def _reward_function(self, current_state, action, next_state, done):
        return 0

    def _updates_per_step(self, step):
        return self.updates_per_step

    def update_q(self, batch):
        q_loss, metrics = self._q_loss(*batch)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        return metrics

    def update_policy(self, batch):
        policy_loss, metrics = self._policy_loss(*batch)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        return metrics

    def _soft_update_target(self):
        for target, online in zip(self.target_q.parameters(), self.online_q.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + online.data * self.tau)

    def __call__(self, env, profiler=None):
        self.generate_random_trajectories(self.replay_buffer, env, self.starting_steps)

        if self.curiosity_pretraining_steps > 0:
            self._train_curiosity_module()
            self._assign_rewards_to_replay_buffer()
            # self._copy_curiosity_features_all_networks()

        print((f'{self.algorithm_name}: training actor / critic'
               f' for {self.training_steps}'))
        current_state = self.start_new_trajectory(env, self.replay_buffer)

        for step in range(self.training_steps):
            # take action, update replay buffer
            action = self.actor.get_action(current_state)
            self.replay_buffer.current_trajectory().actions.append(action)

            if step == 0 and self.suppress_snowball_steps > 0:
                print(('Suppressing throwing snowball for'
                       f' {self.suppress_snowball_steps}'))
            elif step == self.suppress_snowball_steps and step != 0:
                print('No longer suppressing snowball')
            suppressed_snowball = step < self.suppress_snowball_steps \
                and ActionSpace.threw_snowball(current_state, action)
            if suppressed_snowball:
                obs, _, done, _ = env.step(-1)
            else:
                obs, _, done, _ = env.step(action)

            self.replay_buffer.current_trajectory().append_obs(obs)
            self.replay_buffer.current_trajectory().done = done

            # add elements of the transition tuple to the replay buffer individually
            # so we can use replay buffers current state to calculate the reward
            next_state = self.replay_buffer.current_state()

            reward = self._reward_function(current_state, action,
                                           next_state, done)

            self.replay_buffer.current_trajectory().rewards.append(reward)

            self.replay_buffer.increment_step()
            current_state = next_state

            # train models
            if len(self.replay_buffer) > self.batch_size:
                updates_per_step = self._updates_per_step(step)
                step_metrics = None
                for i in range(updates_per_step):
                    batch = self.replay_buffer.sample(batch_size=self.batch_size)
                    metrics = self.train_one_batch(batch)

                    # collect and log metrics:
                    if step_metrics is None:
                        step_metrics = {}
                        for k, v in iter(metrics.items()):
                            step_metrics[k] = [v]
                    else:
                        for k, v in iter(metrics.items()):
                            step_metrics[k].append(v)
                for k, v in iter(step_metrics.items()):
                    step_metrics[k] = sum(v) / updates_per_step
                if self.wandb:
                    wandb.log(
                        {'reward': reward,
                         **step_metrics,
                         'average_its_per_s': self.iteration_rate()},
                        step=self.iter_count)

                if step % self.target_update_interval:
                    self._soft_update_target()

            # save checkpoints, currently just saves actor and gifs
            if self.checkpoint_frequency \
                    and self.iter_count % self.checkpoint_frequency == 0:
                self.save_checkpoint(replay_buffer=self.replay_buffer,
                                     models_with_names=[(self.actor, 'actor'),
                                                        (self.online_q, 'critic')])

            if done:
                print(f'Trajectory completed at iteration {self.iter_count}')
                current_state = self.start_new_trajectory(env, self.replay_buffer)
            elif suppressed_snowball:
                self.replay_buffer.current_trajectory().done = True
                self.replay_buffer.new_trajectory()
                self.replay_buffer.current_trajectory().append_obs(obs)
                current_state = self.replay_buffer.current_state()

            self.log_step()

            if profiler:
                profiler.step()

        print(f'{self.algorithm_name}: Training complete')
        return self.actor, self.replay_buffer

    def train_one_batch(self, batch):
        # load batch onto gpu
        obs, actions, next_obs, done, rewards = batch
        states = ObservationSpace.obs_to_state(obs)
        next_states = ObservationSpace.obs_to_state(next_obs)
        states, next_states = states_to_device((states, next_states), self.device)
        actions = actions.to(self.device)
        done = th.as_tensor(done).unsqueeze(1).float().to(self.device)
        rewards = rewards.float().unsqueeze(1).to(self.device)
        batch = states, actions, next_states, done, rewards

        policy_metrics = self.update_policy(batch)
        q_metrics = self.update_q(batch)
        metrics = {**policy_metrics, **q_metrics}
        return metrics

    def save(self, save_path):
        Path(save_path).mkdir(exist_ok=True)
        self.actor.save(os.path.join(save_path, 'actor.pth'))


class IntrinsicCuriosityTraining(SoftActorCritic):
    def __init__(self, config, actor=None, **kwargs):
        super().__init__(config, actor, **kwargs)
        self.curiosity_pretraining_steps = config.curiosity_pretraining_steps
        self.curiosity_module = CuriosityModule(
            n_observation_frames=config.n_observation_frames).to(self.device)
        self.curiosity_optimizer = th.optim.Adam(self.curiosity_module.parameters(),
                                                 lr=config.curiosity_lr)

    def _reward_function(self, current_state, action, next_state, done):
        reward = self.curiosity_module.reward(current_state, action,
                                              next_state, done)
        return reward

    def _assign_rewards_to_replay_buffer(self):
        print('Calculating rewards for initial steps...')
        reward_dataloader = DataLoader(self.replay_buffer, shuffle=False,
                                       batch_size=self.batch_size, num_workers=4)
        calculated_rewards = []
        for obs, actions, next_obs, done, _reward in reward_dataloader:
            states = ObservationSpace.obs_to_state(obs)
            next_states = ObservationSpace.obs_to_state(next_obs)
            states, next_states = states_to_device((states, next_states), self.device)
            actions = actions.to(self.device)
            done = done.to(self.device)
            rewards = self.curiosity_module.bulk_rewards(states, actions,
                                                         next_states, done)
            calculated_rewards.extend(rewards)
        self.replay_buffer.update_rewards(calculated_rewards)

    def _copy_curiosity_features_all_networks(self):
        self.actor.cnn.load_state_dict(self.curiosity_module.features.state_dict())
        if self.double_q:
            self.online_q._q_network_1.cnn.load_state_dict(
                self.curiosity_module.features.state_dict())
            self.online_q._q_network_2.cnn.load_state_dict(
                self.curiosity_module.features.state_dict())
            self.target_q._q_network_1.cnn.load_state_dict(
                self.curiosity_module.features.state_dict())
            self.target_q._q_network_2.cnn.load_state_dict(
                self.curiosity_module.features.state_dict())
        else:
            self.online_q.cnn.load_state_dict(
                self.curiosity_module.features.state_dict())
            self.target_q.cnn.load_state_dict(
                self.curiosity_module.features.state_dict())

    def update_curiosity(self, batch):
        curiosity_loss, metrics = self.curiosity_module.loss(*batch)
        self.curiosity_optimizer.zero_grad(set_to_none=True)
        curiosity_loss.backward()
        self.curiosity_optimizer.step()
        return metrics

    def train_one_batch(self, batch, curiosity_only=False):
        # load batch onto gpu
        obs, actions, next_obs, done, rewards = batch
        states = ObservationSpace.obs_to_state(obs)
        next_states = ObservationSpace.obs_to_state(next_obs)
        states, next_states = states_to_device((states, next_states), self.device)
        actions = actions.to(self.device)
        done = th.as_tensor(done).unsqueeze(1).float().to(self.device)
        if not curiosity_only:
            rewards = rewards.float().unsqueeze(1).to(self.device)
            mean_reward = rewards.mean().item()
            std_reward = rewards.std().item()
            rewards = (rewards - mean_reward) / std_reward
        batch = states, actions, next_states, done, rewards

        if not curiosity_only:
            policy_metrics = self.update_policy(batch)
            q_metrics = self.update_q(batch)
        else:
            policy_metrics = {}
            q_metrics = {}
        curiosity_metrics = self.update_curiosity(batch)

        metrics = {**policy_metrics, **q_metrics, **curiosity_metrics}
        return metrics

    def _train_curiosity_module(self):
        print((f'Pretraining curiosity module for'
              f' {self.curiosity_pretraining_steps} steps'))
        for step in range(self.curiosity_pretraining_steps):
            batch = self.replay_buffer.sample(batch_size=self.batch_size)
            metrics = self.train_one_batch(batch, curiosity_only=True)
            self.log_step()
            if self.wandb:
                wandb.log(
                    {**metrics,
                     'average_its_per_s': self.iteration_rate()},
                    step=self.iter_count)


class IQLearnSAC(SoftActorCritic):
    def __init__(self, expert_dataset, config, actor=None, **kwargs):
        super().__init__(config, actor, **kwargs)

        self.replay_buffer = MixedReplayBuffer(
            expert_dataset=expert_dataset,
            batch_size=config.batch_size,
            expert_sample_fraction=0.5,
            n_observation_frames=config.n_observation_frames)

        self._q_loss = IQLearnLossSAC(self.online_q, config, target_q=self.target_q)

    def train_one_batch(self, batch):
        expert_batch, replay_batch = batch
        # load batch onto gpu and reconstruct batch with relevant data
        all_states, expert_actions, _replay_actions, expert_done, replay_done = \
            self._q_loss.batches_to_device(expert_batch, replay_batch)

        expert_states, replay_states, expert_next_states, replay_next_states = all_states

        batch_for_q = expert_states, expert_actions, expert_next_states, expert_done, \
            replay_states, _replay_actions, replay_next_states, replay_done
        q_metrics = self.update_q(batch_for_q)

        batch_for_policy = [th.cat(state_component, dim=0) for state_component in
                            zip(expert_states, replay_states)], None, None, None, None
        policy_metrics = self.update_policy(batch_for_policy)

        metrics = {**policy_metrics, **q_metrics}
        return metrics
