from networks.soft_q import SoftQNetwork, TwinnedSoftQNetwork
from algorithms.algorithm import Algorithm
from algorithms.loss_functions.sac import SACQLoss, SACQLossDRQ, \
    SACPolicyLoss, CuriousIQPolicyLoss
from algorithms.loss_functions.iqlearn import IQLearnLossDRQ
from networks.intrinsic_curiosity import CuriosityModule
from helpers.environment import ObservationSpace, ActionSpace
from helpers.datasets import ReplayBuffer, MixedReplayBuffer
from helpers.trajectories import TrajectoryGenerator
from helpers.gpu import disable_gradients, batch_to_device, expert_batch_to_device, \
    cat_batches
from helpers.data_augmentation import DataAugmentation

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import deque
import wandb
import os
from pathlib import Path


class SoftActorCritic(Algorithm):
    def __init__(self, config, actor=None, pretraining=False,
                 initial_replay_buffer=None, initial_iter_count=0):
        super().__init__(config, pretraining=pretraining)
        self.augmentation = DataAugmentation(config)
        method_config = config.pretraining if pretraining else config.method
        self.suppress_snowball_steps = method_config.suppress_snowball_steps
        self.training_steps = method_config.training_steps
        self.batch_size = method_config.batch_size
        self.tau = method_config.tau
        self.double_q = method_config.double_q
        self.drq = method_config.drq
        self.target_update_interval = 1
        self.updates_per_step = 1
        self.curiosity_pretraining_steps = 0
        # Set up replay buffer
        if initial_replay_buffer is None:
            self.replay_buffer = ReplayBuffer(config)
        else:
            self.replay_buffer = initial_replay_buffer
            print((f'Using initial replay buffer'
                   f' with {len(initial_replay_buffer)} steps'))
        self.iter_count += initial_iter_count

        # Set up networks - actor
        if actor is not None:
            self.actor = actor.to(self.device)
        else:
            self.actor = SoftQNetwork(config).to(self.device)

        # Set up networks - critic
        if self.double_q:
            self.online_q = TwinnedSoftQNetwork(config).to(self.device)
            self.target_q = TwinnedSoftQNetwork(config).to(self.device)
        else:
            self.online_q = SoftQNetwork(config).to(self.device)
            self.target_q = SoftQNetwork(config).to(self.device)

        self.target_q.load_state_dict(self.online_q.state_dict())
        disable_gradients(self.target_q)

        # Loss functions
        if self.drq:
            self._q_loss = SACQLossDRQ(self.online_q, self.target_q,
                                       config, pretraining=pretraining)
        else:
            self._q_loss = SACQLoss(self.online_q, self.target_q,
                                    config, pretraining=pretraining)
        self._policy_loss = SACPolicyLoss(self.actor, self.online_q,
                                          config, pretraining=pretraining)

        # Optimizers
        self.policy_optimizer = th.optim.Adam(self.actor.parameters(),
                                              lr=method_config.policy_lr)
        self.q_optimizer = th.optim.Adam(self.online_q.parameters(),
                                         lr=method_config.q_lr)

    def _reward_function(self, current_state, action, next_state, done):
        return 0

    def _updates_per_step(self, step):
        return self.updates_per_step

    def update_q(self, batch, aug_batch=None):
        if self.drq:
            q_loss, metrics = self._q_loss(batch, aug_batch)
        else:
            q_loss, metrics = self._q_loss(batch)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        return metrics

    def update_policy(self, step, batch):
        policy_loss, metrics = self._policy_loss(step, batch)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        return metrics

    def _soft_update_target(self):
        for target, online in zip(self.target_q.parameters(), self.online_q.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + online.data * self.tau)

    def __call__(self, env, profiler=None):
        if self.curiosity_pretraining_steps > 0:
            self._train_curiosity_module()

        print((f'{self.algorithm_name}: training actor / critic'
               f' for {self.training_steps}'))
        current_state = TrajectoryGenerator.new_trajectory(env, self.replay_buffer)

        for step in range(self.training_steps):
            # take action, update replay buffer
            action = self.actor.get_action(current_state)
            self.replay_buffer.current_trajectory().actions.append(action)
            if step == 0 and self.suppress_snowball_steps > 0:
                print(('Suppressing throwing snowball for'
                       f' {min(self.training_steps, self.suppress_snowball_steps)}'))
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
                    metrics = self.train_one_batch(step, batch)

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
                current_state = TrajectoryGenerator.new_trajectory(env,
                                                                   self.replay_buffer)
            elif suppressed_snowball:
                current_state = TrajectoryGenerator.new_trajectory(env,
                                                                   self.replay_buffer,
                                                                   reset_env=False,
                                                                   current_obs=obs)

            self.log_step()

            if profiler:
                profiler.step()

        print(f'{self.algorithm_name}: Training complete')
        return self.actor, self.replay_buffer

    def train_one_batch(self, step, batch):
        # load batch onto gpu
        batch = batch_to_device(batch)
        aug_batch = self.augmentation(batch)
        if self.drq:
            q_metrics = self.update_q(batch, aug_batch)
        else:
            q_metrics = self.update_q(aug_batch)
        policy_metrics = self.update_policy(step, aug_batch)
        metrics = {**policy_metrics, **q_metrics}
        return metrics

    def save(self, save_path):
        Path(save_path).mkdir(exist_ok=True)
        self.actor.save(os.path.join(save_path, 'actor.pth'))


class IntrinsicCuriosityTraining(SoftActorCritic):
    def __init__(self, config, actor=None, pretraining=True, **kwargs):
        super().__init__(config, actor, **kwargs)
        method_config = config.pretraining if pretraining else config.method
        self.curiosity_pretraining_steps = method_config.curiosity_pretraining_steps
        self.normalize_reward = method_config.normalize_reward
        self.recent_rewards = deque(maxlen=1000)
        self.curiosity_module = CuriosityModule(
            n_observation_frames=config.n_observation_frames).to(self.device)
        self.curiosity_optimizer = th.optim.Adam(self.curiosity_module.parameters(),
                                                 lr=method_config.curiosity_lr)

    def _reward_function(self, current_state, action, next_state, done):
        reward = self.curiosity_module.reward(current_state, action,
                                              next_state, done)
        if not self.normalize_reward:
            return reward
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 10 and np.std(self.recent_rewards) != 0:
            mean = sum(self.recent_rewards) / len(self.recent_rewards)
            std = np.std(self.recent_rewards)
            relative_reward = (reward - mean) / std * self.curiosity_module.eta
        else:
            relative_reward = 0
        relative_reward = min(max(relative_reward, -1), 1)
        return relative_reward

    def update_curiosity(self, batch):
        curiosity_loss, metrics = self.curiosity_module.loss(batch)
        self.curiosity_optimizer.zero_grad(set_to_none=True)
        curiosity_loss.backward()
        self.curiosity_optimizer.step()
        return metrics

    def train_one_batch(self, step, batch, curiosity_only=False):
        batch = batch_to_device(batch)
        aug_batch = self.augmentation(batch)
        if not curiosity_only:
            if self.drq:
                q_metrics = self.update_q(batch, aug_batch)
            else:
                q_metrics = self.update_q(aug_batch)
            policy_metrics = self.update_policy(step, aug_batch)
        else:
            policy_metrics = {}
            q_metrics = {}
        curiosity_metrics = self.update_curiosity(aug_batch)

        metrics = {**policy_metrics, **q_metrics, **curiosity_metrics}
        return metrics

    def _train_curiosity_module(self):
        print((f'Pretraining curiosity module for'
              f' {self.curiosity_pretraining_steps} steps'))
        for step in range(self.curiosity_pretraining_steps):
            batch = self.replay_buffer.sample(batch_size=self.batch_size)
            metrics = self.train_one_batch(step, batch, curiosity_only=True)
            self.log_step()
            if self.wandb:
                wandb.log(
                    {**metrics,
                     'average_its_per_s': self.iteration_rate()},
                    step=self.iter_count)


class CuriousIQ(IntrinsicCuriosityTraining):
    def __init__(self, expert_dataset, config, **kwargs):
        super().__init__(config, pretraining=False, **kwargs)
        self.curiosity_reward = config.method.curiosity_reward
        self.online_curiosity_training = config.method.online_curiosity_training
        self.initial_curiosity_fraction = config.method.initial_curiosity_fraction
        self.iqlearn_q = SoftQNetwork(config).to(self.device)
        if config.method.target_q:
            self.iqlearn_target_q = SoftQNetwork(config).to(self.device)
            self.iqlearn_target_q.load_state_dict(self.iqlearn_q.state_dict())
            disable_gradients(self.iqlearn_target_q)
            print('IQLearn Target Network Initialized')
        else:
            self.iqlearn_target_q = None
        if self.drq:
            self._iqlearn_loss = IQLearnLossDRQ(self.iqlearn_q, config,
                                                target_q=self.iqlearn_target_q)
        else:
            self._iqlearn_loss = IQLearnLoss(self.iqlearn_q, config,
                                             target_q=self.iqlearn_target_q)
        self.iqlearn_optimizer = th.optim.Adam(self.iqlearn_q.parameters(),
                                               lr=config.method.iqlearn_lr)
        self._policy_loss = CuriousIQPolicyLoss(self.actor, self.online_q, self.iqlearn_q,
                                                config)
        self.replay_buffer = MixedReplayBuffer(
            expert_dataset=expert_dataset,
            config=config,
            batch_size=config.method.batch_size,
            initial_replay_buffer=self.replay_buffer)

    def _soft_update_target(self):
        if self.curiosity_reward:
            for target, online in zip(
                    self.target_q.parameters(), self.online_q.parameters()):
                target.data.copy_(target.data * (1.0 - self.tau) + online.data * self.tau)
        if self.iqlearn_target_q is None:
            return
        for target, online in zip(self.iqlearn_target_q.parameters(),
                                  self.iqlearn_q.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + online.data * self.tau)

    def update_iqlearn(self, expert_batch, replay_batch,
                       expert_batch_aug=None, replay_batch_aug=None):
        loss, metrics = self._iqlearn_loss(expert_batch, replay_batch,
                                           expert_batch_aug, replay_batch_aug)
        self.iqlearn_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.iqlearn_optimizer.step()
        return metrics

    def train_one_batch(self, step, batch, curiosity_only=False):
        expert_batch, replay_batch = batch
        expert_batch = expert_batch_to_device(expert_batch)
        replay_batch = batch_to_device(replay_batch)
        expert_batch_aug = self.augmentation(expert_batch)
        replay_batch_aug = self.augmentation(replay_batch)
        combined_batch = cat_batches((expert_batch, replay_batch,
                                     expert_batch_aug, replay_batch_aug))
        if not curiosity_only:
            if self.drq:
                q_metrics = self.update_q(replay_batch, replay_batch_aug)
                iqlearn_metrics = self.update_iqlearn(expert_batch, replay_batch,
                                                      expert_batch_aug, replay_batch_aug)
            else:
                q_metrics = self.update_q(replay_batch_aug)
                iqlearn_metrics = self.update_iqlearn(expert_batch_aug, replay_batch_aug)
            policy_metrics = self.update_policy(step, combined_batch)
        else:
            q_metrics = {}
            iqlearn_metrics = {}
            policy_metrics = {}
        if curiosity_only or \
                (self.curiosity_reward and self.online_curiosity_training and
                 step < (self.config.method.curiosity_only_steps
                         + self.config.method.curiosity_fade_out_steps)):
            curiosity_metrics = self.update_curiosity(combined_batch)
        else:
            curiosity_metrics = {}

        metrics = {**policy_metrics, **q_metrics, **curiosity_metrics, **iqlearn_metrics}
        return metrics

    def _reward_function(self, current_state, action, next_state, done):
        if self.curiosity_reward:
            reward = super()._reward_function(current_state, action, next_state, done)
        else:
            reward = 0
        return reward
