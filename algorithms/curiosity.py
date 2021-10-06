from networks.soft_q import SoftQNetwork
from algorithms.sac import SoftActorCritic
from algorithms.loss_functions.sac import CuriousIQPolicyLoss
from algorithms.loss_functions.iqlearn import IQLearnLossDRQ
from networks.intrinsic_curiosity import CuriosityModule
from utils.environment import ObservationSpace, ActionSpace
from utils.datasets import MixedReplayBuffer, MixedSegmentReplayBuffer
from utils.gpu import disable_gradients, cat_batches

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import deque
import wandb


class IntrinsicCuriosityTraining(SoftActorCritic):
    def __init__(self, config, actor=None, pretraining=True, **kwargs):
        super().__init__(config, actor, **kwargs)
        method_config = config.pretraining if pretraining else config.method
        self.curiosity_pretraining_steps = method_config.curiosity_pretraining_steps
        self.normalize_reward = method_config.normalize_reward
        self.running_avg_loss = deque(maxlen=250)
        self.curiosity_module = CuriosityModule(config).to(self.device)
        self.curiosity_optimizer = th.optim.Adam(self.curiosity_module.parameters(),
                                                 lr=method_config.curiosity_lr)

    def __call__(self, env, profiler=None):
        if self.curiosity_pretraining_steps > 0:
            self._train_curiosity_module()
            self._assign_rewards_to_replay_buffer()
        super().__call__(env, profiler)

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
        batch = self.gpu_loader.batch_to_device(batch)
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
            replay_batch = self.gpu_loader.batch_to_device(replay_batch)
            rewards = self.curiosity_module.bulk_rewards(replay_batch, expert=False)
            if isinstance(rewards, (int, float)):
                rewards = [rewards]
            replay_rewards.extend(rewards)

        expert_dataloader = DataLoader(self.replay_buffer.expert_dataset, shuffle=False,
                                       batch_size=self.batch_size, num_workers=4)
        expert_rewards = []
        for expert_batch in expert_dataloader:
            expert_batch = self.gpu_loader.expert_batch_to_device(expert_batch)
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

        kwargs = dict(
            expert_dataset=expert_dataset,
            config=config,
            batch_size=config.method.batch_size,
            initial_replay_buffer=self.replay_buffer)
        if self.config.lstm_layers == 0:
            self.replay_buffer = MixedReplayBuffer(**kwargs)
        else:
            self.replay_buffer = MixedSegmentReplayBuffer(**kwargs)

    def update_model_alphas(self):
        alpha = self._policy_loss.log_alpha.detach().exp().item()
        self.online_q.alpha = alpha
        self.target_q.alpha = alpha
        self.actor.alpha = alpha
        self.iqlearn_q.alpha = alpha
        self.iqlearn_target_q.alpha = alpha

    def _soft_update_target(self):
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
        expert_batch = self.gpu_loader.expert_batch_to_device(expert_batch)
        replay_batch = self.gpu_loader.batch_to_device(replay_batch)
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
                (self.online_curiosity_training and
                 step < (self.config.method.curiosity_only_steps
                         + self.config.method.curiosity_fade_out_steps)):
            curiosity_metrics = self.update_curiosity(combined_batch)
        else:
            curiosity_metrics = {}

        metrics = {**policy_metrics, **q_metrics, **curiosity_metrics, **iqlearn_metrics}
        return metrics
