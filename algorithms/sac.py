from networks.soft_q import SoftQNetwork, TwinnedSoftQNetwork
from algorithms.algorithm import Algorithm
from algorithms.loss_functions.sac import SACQLoss, SACQLossDRQ, \
    SACPolicyLoss, CuriousIQPolicyLoss
from algorithms.loss_functions.iqlearn import IQLearnLossDRQ
from networks.intrinsic_curiosity import CuriosityModule
from helpers.environment import ObservationSpace, ActionSpace
from helpers.datasets import ReplayBuffer, MixedReplayBuffer
from helpers.trajectories import TrajectoryGenerator
from helpers.gpu import disable_gradients, cat_batches
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
        self.entropy_lr = method_config.entropy_lr
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

        self.entropy_adjustment = method_config.entropy_adjustment
        self.target_entropy_ratio = method_config.target_entropy_ratio
        if self.entropy_adjustment:
            self.initialize_alpha_optimization()

        # Optimizers
        self.policy_optimizer = th.optim.Adam(self.actor.parameters(),
                                              lr=method_config.policy_lr)
        self.q_optimizer = th.optim.Adam(self.online_q.parameters(),
                                         lr=method_config.q_lr)

    def update_model_alphas(self):
        alpha = self._policy_loss.log_alpha.detach().exp()
        self.online_q.alpha = alpha
        self.target_q.alpha = alpha
        self.actor.alpha = alpha

    def initialize_alpha_optimization(self):
        self._policy_loss.target_entropy = \
            -np.log(1.0 / len(ActionSpace.actions())) * self.target_entropy_ratio
        print('Target entropy: ', self._policy_loss.target_entropy)
        self._policy_loss.log_alpha = th.zeros(1, device=self.device, requires_grad=True)
        self.alpha_optimizer = th.optim.Adam([self._policy_loss.log_alpha],
                                             lr=self.entropy_lr)

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
        policy_loss, alpha_loss, metrics = self._policy_loss(step, batch)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        if self.entropy_adjustment:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.update_model_alphas()
            metrics['alpha'] = self.actor.alpha
        return metrics

    def _soft_update_target(self):
        for target, online in zip(self.target_q.parameters(), self.online_q.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + online.data * self.tau)

    def __call__(self, env, profiler=None):
        if self.curiosity_pretraining_steps > 0:
            self._train_curiosity_module()
            self._assign_rewards_to_replay_buffer()

        print((f'{self.algorithm_name}: training actor / critic'
               f' for {self.training_steps}'))

        rewards_window = deque(maxlen=10)  # last N rewards
        steps_window = deque(maxlen=10)  # last N episode steps

        episode_reward = 0
        episode_steps = 0

        current_state = TrajectoryGenerator.new_trajectory(
            env, self.replay_buffer, initial_hidden=self.actor.initial_hidden())

        for step in range(self.training_steps):
            # take action, update replay buffer
            action, hidden = self.actor.get_action(current_state)
            self.replay_buffer.current_trajectory().actions.append(action)
            if step == 0 and self.suppress_snowball_steps > 0:
                print(('Suppressing throwing snowball for'
                       f' {min(self.training_steps, self.suppress_snowball_steps)}'))
            elif step == self.suppress_snowball_steps and step != 0:
                print('No longer suppressing snowball')
            suppressed_snowball = step < self.suppress_snowball_steps \
                and ActionSpace.threw_snowball(current_state, action)
            if suppressed_snowball:
                obs, r, done, _ = env.step(-1)
            else:
                obs, r, done, _ = env.step(action)

            self.replay_buffer.current_trajectory().append_obs(obs, hidden)
            self.replay_buffer.current_trajectory().done = done

            # add elements of the transition tuple to the replay buffer individually
            # so we can use replay buffers current state to calculate the reward
            next_state = self.replay_buffer.current_state()

            reward = self._reward_function(current_state, action,
                                           next_state, done)
            self.replay_buffer.current_trajectory().rewards.append(reward)

            self.replay_buffer.increment_step()
            episode_reward += r
            episode_steps += 1

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

            if done or suppressed_snowball:
                print(f'Trajectory completed at iteration {self.iter_count}')
                if suppressed_snowball:
                    print('Suppressed Snowball')
                    reset_env = False
                else:
                    reset_env = True
                TrajectoryGenerator.new_trajectory(
                    env, self.replay_buffer,
                    reset_env=reset_env, current_obs=obs,
                    initial_hidden=self.actor.initial_hidden())

                rewards_window.append(episode_reward)
                steps_window.append(episode_steps)
                if self.wandb:
                    wandb.log({'Rewards/train_reward': np.mean(rewards_window)},
                              step=self.iter_count)
                    wandb.log({'Timesteps/train': np.mean(steps_window)},
                              step=self.iter_count)

                episode_reward = 0
                episode_steps = 0

            self.log_step()

            if profiler:
                profiler.step()

        print(f'{self.algorithm_name}: Training complete')
        return self.actor, self.replay_buffer

    def train_one_batch(self, step, batch):
        # load batch onto gpu
        batch = self.actor.gpu_loader.batch_to_device(batch)
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
        self.running_avg_loss = deque(maxlen=250)
        self.curiosity_module = CuriosityModule(
            n_observation_frames=config.n_observation_frames).to(self.device)
        self.curiosity_optimizer = th.optim.Adam(self.curiosity_module.parameters(),
                                                 lr=method_config.curiosity_lr)

    def _reward_function(self, current_state, action, next_state, done):
        reward = self.curiosity_module.reward(current_state, action,
                                              next_state, done)
        if not self.normalize_reward:
            return reward
        if len(self.running_avg_loss) > 10:
            relative_reward = (reward - self.reward_mean) / self.reward_std \
                * self.curiosity_module.eta
        else:
            relative_reward = 0
        relative_reward = min(max(relative_reward, -1), 1)
        return relative_reward

    def update_curiosity(self, batch):
        curiosity_loss, metrics = self.curiosity_module.loss(batch)
        self.curiosity_optimizer.zero_grad(set_to_none=True)
        curiosity_loss.backward()
        self.curiosity_optimizer.step()
        self.running_avg_loss.append(metrics['curiosity_loss_total'])
        return metrics

    def train_one_batch(self, step, batch, curiosity_only=False):
        batch = self.actor.gpu_loader.batch_to_device(batch)
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

    def _assign_rewards_to_replay_buffer(self):
        print('Calculating rewards for initial steps...')
        avg_loss = sum(self.running_avg_loss) / len(self.running_avg_loss)
        replay_dataloader = DataLoader(self.replay_buffer, shuffle=False,
                                       batch_size=self.batch_size, num_workers=4)
        replay_rewards = []
        for replay_batch in replay_dataloader:
            replay_batch = self.actor.gpu_loader.batch_to_device(replay_batch)
            rewards = self.curiosity_module.bulk_rewards(replay_batch, expert=False)
            if isinstance(rewards, (int, float)):
                rewards = [rewards]
            replay_rewards.extend(rewards)

        expert_dataloader = DataLoader(self.replay_buffer.expert_dataset, shuffle=False,
                                       batch_size=self.batch_size, num_workers=4)
        expert_rewards = []
        for expert_batch in expert_dataloader:
            expert_batch = self.actor.gpu_loader.expert_batch_to_device(expert_batch)
            rewards = self.curiosity_module.bulk_rewards(expert_batch, expert=True)
            if isinstance(rewards, (int, float)):
                rewards = [rewards]
            expert_rewards.extend(rewards)

        all_rewards = np.array([reward for reward in replay_rewards + expert_rewards
                                if reward != 1 and reward != 0])
        replay_reward_curiosity = np.array([reward for reward in replay_rewards
                                            if reward != 1 and reward != 0])
        expert_reward_curiosity = np.array([reward for reward in expert_rewards
                                            if reward != 1 and reward != 0])
        self.reward_mean = np.median(all_rewards)
        self.reward_std = all_rewards.std()

        replay_rewards = [max(min((reward - np.median(replay_reward_curiosity))
                                  / replay_reward_curiosity.std()
                                  * self.curiosity_module.eta, 1), -1)
                          for reward in replay_rewards]
        expert_rewards = [max(min((reward - np.median(expert_reward_curiosity))
                                  / expert_reward_curiosity.std()
                                  * self.curiosity_module.eta, 1), -1)
                          for reward in expert_rewards]
        if self.wandb:
            wandb.log({'CuriosityRewards/all_rewards': all_rewards}, step=self.iter_count)
            wandb.log({'CuriosityRewards/replay_rewards': np.array(replay_rewards)},
                      step=self.iter_count)
            wandb.log({'CuriosityRewards/expert_rewards': np.array(expert_rewards)},
                      step=self.iter_count)
        self.replay_buffer.update_rewards(replay_rewards, expert_rewards)


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
        if self.entropy_adjustment:
            self.initialize_alpha_optimization()

        self.replay_buffer = MixedReplayBuffer(
            expert_dataset=expert_dataset,
            config=config,
            batch_size=config.method.batch_size,
            initial_replay_buffer=self.replay_buffer)

    def update_model_alphas(self):
        alpha = self._policy_loss.log_alpha.detach().exp().item()
        self.online_q.alpha = alpha
        self.target_q.alpha = alpha
        self.actor.alpha = alpha
        self.iqlearn_q.alpha = alpha
        self.iqlearn_target_q.alpha = alpha

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
        expert_batch = self.actor.gpu_loader.expert_batch_to_device(expert_batch)
        replay_batch = self.actor.gpu_loader.batch_to_device(replay_batch)
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
