from networks.soft_q import SoftQNetwork, TwinnedSoftQNetwork
from core.algorithm import Algorithm
from algorithms.loss_functions.sac import SACQLoss, SACQLossDRQ, \
    SACPolicyLoss, CuriousIQPolicyLoss
from algorithms.loss_functions.iqlearn import IQLearnLossDRQ
from networks.intrinsic_curiosity import CuriosityModule
from core.environment import ObservationSpace, ActionSpace
from core.datasets import ReplayBuffer, MixedReplayBuffer
from core.trajectories import TrajectoryGenerator
from core.gpu import disable_gradients, cat_batches
from core.data_augmentation import DataAugmentation

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.core.data import DataLoader

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
        self.training_steps = int(method_config.training_steps)
        self.batch_size = method_config.batch_size
        self.tau = method_config.tau
        self.entropy_lr = method_config.entropy_lr
        # self.double_q = method_config.double_q
        self.drq = method_config.drq
        self.target_update_interval = 1
        self.updates_per_step = 1
        self.initial_alpha = config.alpha
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
        # if self.double_q:
        #     self.online_q = TwinnedSoftQNetwork(config).to(self.device)
        #     self.target_q = TwinnedSoftQNetwork(config).to(self.device)
        # else:
        self.online_q = SoftQNetwork(config).to(self.device)
        self.target_q = SoftQNetwork(config).to(self.device)
        self.target_q.load_state_dict(self.online_q.state_dict())
        disable_gradients(self.target_q)

        # Loss functions
        self.initialize_loss_functions()

        self.entropy_tuning = method_config.entropy_tuning
        self.target_entropy_ratio = method_config.target_entropy_ratio
        if self.entropy_tuning:
            self.initialize_alpha_optimization()

        # Optimizers
        self.policy_optimizer = th.optim.Adam(self.actor.parameters(),
                                              lr=method_config.policy_lr)
        self.q_optimizer = th.optim.Adam(self.online_q.parameters(),
                                         lr=method_config.q_lr)

    def initialize_loss_functions(self):
        if self.drq:
            self._q_loss = SACQLossDRQ(self.online_q, self.target_q, self.config)
        else:
            self._q_loss = SACQLoss(self.online_q, self.target_q, self.config)
        self._policy_loss = SACPolicyLoss(self.actor, self.online_q, self.config)

    def initialize_alpha_optimization(self):
        self._policy_loss.target_entropy = \
            -np.log(1.0 / len(ActionSpace.actions())) * self.target_entropy_ratio
        print('Target entropy: ', self._policy_loss.target_entropy)
        self._policy_loss.log_alpha = th.tensor(np.log(self.initial_alpha),
                                                device=self.device, requires_grad=True)
        self.alpha_optimizer = th.optim.Adam([self._policy_loss.log_alpha],
                                             lr=self.entropy_lr)

    def update_model_alphas(self):
        alpha = self._policy_loss.log_alpha.detach().exp()
        self.online_q.alpha = alpha
        self.target_q.alpha = alpha
        self.actor.alpha = alpha

    def _reward_function(self, current_state, action, next_state, done):
        return 0

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
        policy_loss, alpha_loss, final_hidden, metrics = self._policy_loss(step, batch)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        if self.entropy_tuning:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.update_model_alphas()
            metrics['alpha'] = self.actor.alpha
        return metrics, final_hidden

    def _soft_update_target(self):
        for target, online in zip(self.target_q.parameters(), self.online_q.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + online.data * self.tau)

    def train_one_batch(self, step, batch):
        batch, batch_idx = batch
        batch = self.gpu_loader.batch_to_device(batch)
        aug_batch = self.augmentation(batch)
        if self.drq:
            q_metrics = self.update_q(batch, aug_batch)
        else:
            q_metrics = self.update_q(aug_batch)
        policy_metrics, final_hidden = self.update_policy(step, aug_batch)
        if final_hidden is not None:
            self.replay_buffer.update_hidden(batch_idx, final_hidden)
        metrics = {**policy_metrics, **q_metrics}
        return metrics

    def __call__(self, env, profiler=None):
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
            action, hidden = self.actor.get_action(
                self.gpu_loader.state_to_device(current_state))
            self.replay_buffer.current_trajectory().actions.append(action)

            suppressed_snowball = self.suppressed_snowball(step, current_state, action)
            if suppressed_snowball:
                obs, r, done, _ = env.step(-1)
            else:
                obs, r, done, _ = env.step(action)
            if self.wandb:
                wandb.log({'Rewards/ground_truth_reward': r}, step=self.iter_count)

            self.replay_buffer.current_trajectory().append_obs(obs, hidden)
            self.replay_buffer.current_trajectory().done = done

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
                step_metrics = None
                for i in range(self.updates_per_step):
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
                    step_metrics[k] = sum(v) / len(v)
                if self.wandb:
                    wandb.log(
                        {'reward': reward,
                         **step_metrics,
                         'average_its_per_s': self.iteration_rate()},
                        step=self.iter_count)

                if step % self.target_update_interval:
                    self._soft_update_target()

            # save checkpoints, currently just saves gifs
            if self.checkpoint_frequency \
                    and self.iter_count % self.checkpoint_frequency == 0:
                self.save_checkpoint(replay_buffer=self.replay_buffer)

            if done or suppressed_snowball \
                    or len(self.replay_buffer.current_trajectory()) == \
                    self.max_training_episode_length:
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
