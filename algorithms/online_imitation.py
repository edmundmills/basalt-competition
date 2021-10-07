from algorithms.algorithm import Algorithm
from algorithms.loss_functions.iqlearn import IQLearnLoss, IQLearnLossDRQ
# from algorithms.loss_functions.sqil import SqilLoss
from utils.environment import ObservationSpace, ActionSpace
from utils.datasets import MixedReplayBuffer, MixedSegmentReplayBuffer
from utils.data_augmentation import DataAugmentation
from utils.trajectories import TrajectoryGenerator

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from collections import deque

import wandb
import os


class OnlineImitation(Algorithm):
    def __init__(self, expert_dataset, model, config,
                 initial_replay_buffer=None, initial_iter_count=0):
        super().__init__(config)
        self.lr = config.method.learning_rate
        self.suppress_snowball_steps = config.method.suppress_snowball_steps
        self.training_steps = config.method.training_steps
        self.batch_size = config.method.batch_size
        self.model = model
        self.expert_dataset = expert_dataset
        self.initial_replay_buffer = initial_replay_buffer
        self.iter_count += initial_iter_count
        self.drq = config.method.drq
        self.augmentation = DataAugmentation(config)
        self.initialize_loss_function(model, config)

    def initialize_loss_function(self, model, config):
        if config.method.loss_function == 'sqil':
            self.loss_function = SqilLoss(model, config)
        elif config.method.loss_function == 'iqlearn' and self.drq:
            self.loss_function = IQLearnLossDRQ(model, config)
        elif config.method.loss_function == 'iqlearn':
            self.loss_function = IQLearnLoss(model, config)

    def initialize_replay_buffer(self):
        initial_replay_buffer = self.initial_replay_buffer
        if initial_replay_buffer is not None:
            print((f'Using initial replay buffer'
                   f' with {len(initial_replay_buffer)} steps'))
        kwargs = dict(
            expert_dataset=self.expert_dataset,
            config=self.config,
            batch_size=self.batch_size,
            initial_replay_buffer=initial_replay_buffer
        )
        if self.config.lstm_layers == 0:
            replay_buffer = MixedReplayBuffer(**kwargs)
        else:
            replay_buffer = MixedSegmentReplayBuffer(**kwargs)
        return replay_buffer

    def train_one_batch(self, batch):
        (expert_batch, expert_idx), (replay_batch, replay_idx) = batch
        expert_batch, replay_batch = self.gpu_loader.batches_to_device(
            expert_batch, replay_batch)
        aug_expert_batch = self.augmentation(expert_batch)
        aug_replay_batch = self.augmentation(replay_batch)
        if self.drq:
            loss, metrics = self.loss_function(expert_batch, replay_batch,
                                               aug_expert_batch, aug_replay_batch)
        else:
            loss, metrics = self.loss_function(aug_expert_batch, aug_replay_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.wandb:
            wandb.log(metrics, step=self.iter_count)

    def __call__(self, env, profiler=None):
        model = self.model
        expert_dataset = self.expert_dataset

        self.optimizer = th.optim.Adam(model.parameters(),
                                       lr=self.lr)

        replay_buffer = self.initialize_replay_buffer()

        TrajectoryGenerator.new_trajectory(env, replay_buffer,
                                           initial_hidden=model.initial_hidden())

        print((f'{self.algorithm_name}: Starting training'
               f' for {self.training_steps} steps (iteration {self.iter_count})'))
        rewards_window = deque(maxlen=10)  # last N rewards
        steps_window = deque(maxlen=10)  # last N episode steps

        episode_reward = 0
        episode_steps = 0

        for step in range(self.training_steps):
            current_state = replay_buffer.current_state()
            action, hidden = model.get_action(
                self.gpu_loader.state_to_device(current_state))
            if step == 0 and self.suppress_snowball_steps > 0:
                print(('Suppressing throwing snowball for'
                       f' {min(self.training_steps, self.suppress_snowball_steps)}'
                       ' steps'))
            elif step == self.suppress_snowball_steps and step != 0:
                print('No longer suppressing snowball')
            suppressed_snowball = step < self.suppress_snowball_steps \
                and ActionSpace.threw_snowball(current_state, action)
            if suppressed_snowball:
                next_obs, r, done, _ = env.step(-1)
            else:
                next_obs, r, done, _ = env.step(action)
            if self.wandb:
                wandb.log({'Rewards/ground_truth_reward': r}, step=self.iter_count)
                if 'compass' in next_obs.keys():
                    wandb.log({'compass': next_obs['compass']['angle']},
                              step=self.iter_count)

            episode_reward += r
            episode_steps += 1

            replay_buffer.append_step(action, hidden, r, next_obs, done)

            if len(replay_buffer) >= replay_buffer.replay_batch_size:
                self.train_one_batch(replay_buffer.sample(batch_size=self.batch_size))

            self.log_step()

            if self.checkpoint_frequency and \
                    self.iter_count % self.checkpoint_frequency == 0:
                self.save_checkpoint(replay_buffer=replay_buffer,
                                     models_with_names=[(model, 'model')])

            if done or suppressed_snowball \
                    or len(replay_buffer.current_trajectory()) == \
                    self.max_training_episode_length:
                print(f'Trajectory completed at iteration {self.iter_count}')
                if suppressed_snowball:
                    print('Suppressed Snowball')
                    reset_env = False
                else:
                    reset_env = True
                TrajectoryGenerator.new_trajectory(
                    env, replay_buffer,
                    reset_env=reset_env, current_obs=next_obs,
                    initial_hidden=model.initial_hidden())

                rewards_window.append(episode_reward)
                steps_window.append(episode_steps)
                if self.wandb:
                    wandb.log({'Rewards/train_reward': np.mean(rewards_window)},
                              step=self.iter_count)
                    wandb.log({'Timesteps/train': np.mean(steps_window)},
                              step=self.iter_count)

                episode_reward = 0
                episode_steps = 0

            if profiler:
                profiler.step()

        print(f'{self.algorithm_name}: Training complete')
        return model, replay_buffer
