from algorithms.algorithm import Algorithm
from algorithms.loss_functions.iqlearn import IQLearnLoss, IQLearnLossDRQ
from algorithms.loss_functions.sqil import SqilLoss
from helpers.environment import ObservationSpace, ActionSpace
from helpers.datasets import MixedReplayBuffer
from helpers.gpu import batches_to_device
from helpers.data_augmentation import DataAugmentation
from helpers.trajectories import TrajectoryGenerator

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
        if config.method.loss_function == 'sqil':
            self.loss_function = SqilLoss(model, config)
        elif config.method.loss_function == 'iqlearn' and self.drq:
            self.loss_function = IQLearnLossDRQ(model, config)
        elif config.method.loss_function == 'iqlearn':
            self.loss_function = IQLearnLoss(model, config)

    def __call__(self, env, profiler=None):
        model = self.model
        expert_dataset = self.expert_dataset
        initial_replay_buffer = self.initial_replay_buffer

        optimizer = th.optim.Adam(model.parameters(),
                                  lr=self.lr)

        if initial_replay_buffer is not None:
            print((f'Using initial replay buffer'
                   f' with {len(initial_replay_buffer)} steps'))
        replay_buffer = MixedReplayBuffer(
            expert_dataset=expert_dataset,
            config=self.config,
            batch_size=self.batch_size,
            initial_replay_buffer=initial_replay_buffer)

        TrajectoryGenerator.new_trajectory(env, replay_buffer)

        print((f'{self.algorithm_name}: Starting training'
               f' for {self.training_steps} steps (iteration {self.iter_count})'))
        rewards_window = deque(maxlen=10)  # last N rewards
        steps_window = deque(maxlen=10)  # last N episode steps

        episode_reward = 0
        episode_steps = 0

        for step in range(self.training_steps):
            current_state = replay_buffer.current_state()
            action = model.get_action(current_state)
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

            episode_reward += r
            episode_steps += 1

            replay_buffer.append_step(action, r, next_obs, done)

            if len(replay_buffer) >= replay_buffer.replay_batch_size:
                expert_batch = replay_buffer.sample_expert()
                replay_batch = replay_buffer.sample_replay()
                expert_batch, replay_batch = batches_to_device(expert_batch, replay_batch)
                aug_expert_batch = self.augmentation(expert_batch)
                aug_replay_batch = self.augmentation(replay_batch)
                if self.drq:
                    loss, metrics = self.loss_function(expert_batch, replay_batch,
                                                       aug_expert_batch, aug_replay_batch)
                else:
                    loss, metrics = self.loss_function(aug_expert_batch, aug_replay_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if self.wandb:
                    wandb.log(metrics, step=self.iter_count)

            self.log_step()

            if self.checkpoint_frequency and \
                    self.iter_count % self.checkpoint_frequency == 0:
                self.save_checkpoint(replay_buffer=replay_buffer,
                                     models_with_names=[(model, 'model')])

            if done or suppressed_snowball:
                print(f'Trajectory completed at iteration {self.iter_count}')
                if suppressed_snowball:
                    print('Suppressed Snowball')
                    reset_env = False
                else:
                    reset_env = True
                TrajectoryGenerator.new_trajectory(env, replay_buffer,
                                                   reset_env=reset_env,
                                                   current_obs=next_obs)

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
