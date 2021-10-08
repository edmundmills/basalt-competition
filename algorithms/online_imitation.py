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
import time

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
        self.curriculum_training = config.curriculum_training
        self.initial_curriculum_size = config.initial_curriculum_size
        self.curriculum_fraction_of_training = config.curriculum_fraction_of_training
        self.augmentation = DataAugmentation(config)
        self.cyclic_learning_rate = config.cyclic_learning_rate
        self.initialize_loss_function(model, config)

    def max_episode_length(self, step):
        if not self.curriculum_training:
            return self.max_training_episode_length
        curriculum_fraction = step / (self.curriculum_fraction_of_training
                                      * self.training_steps)
        curriculum_span = self.max_training_episode_length - self.initial_curriculum_size
        curriculum_length = int(self.initial_curriculum_size +
                                curriculum_fraction * curriculum_span)
        if curriculum_length >= self.max_training_episode_length:
            return 100000
        else:
            return curriculum_length

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
        self.replay_buffer = replay_buffer
        return replay_buffer

    def train_one_batch(self, batch):
        (expert_batch, expert_idx), (replay_batch, replay_idx) = batch
        expert_batch, replay_batch = self.gpu_loader.batches_to_device(
            expert_batch, replay_batch)
        aug_expert_batch = self.augmentation(expert_batch)
        aug_replay_batch = self.augmentation(replay_batch)
        if self.drq:
            loss, metrics, final_hidden = self.loss_function(
                expert_batch, replay_batch, aug_expert_batch, aug_replay_batch)
        else:
            loss, metrics, final_hidden = self.loss_function(
                aug_expert_batch, aug_replay_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.cyclic_learning_rate:
            self.scheduler.step()
            metrics['learning_rate'] = self.scheduler.get_last_lr()[0]

        if final_hidden is not None:
            final_hidden_expert, final_hidden_replay = final_hidden.chunk(2, dim=0)
            self.replay_buffer.update_hidden(replay_idx, final_hidden_replay,
                                             expert_idx, final_hidden_expert)

        if self.wandb:
            wandb.log(metrics, step=self.iter_count)

    def __call__(self, env, profiler=None):
        model = self.model
        expert_dataset = self.expert_dataset

        self.optimizer = th.optim.AdamW(model.parameters(), lr=self.lr)
        if self.cyclic_learning_rate:
            decay_factor = .25**(1/(self.training_steps/2))
            self.scheduler = th.optim.lr_scheduler.CyclicLR(self.optimizer,
                                                            base_lr=self.lr,
                                                            max_lr=self.lr*10,
                                                            mode='exp_range',
                                                            gamma=decay_factor,
                                                            step_size_up=2000,
                                                            cycle_momentum=False)

        replay_buffer = self.initialize_replay_buffer()

        TrajectoryGenerator.new_trajectory(env, replay_buffer,
                                           initial_hidden=model.initial_hidden())

        print((f'{self.algorithm_name}: Starting training'
               f' for {self.training_steps} steps (iteration {self.iter_count})'))

        for step in range(self.training_steps):
            current_max_episode_length = self.max_episode_length(step)
            current_state = replay_buffer.current_state()
            action, hidden = model.get_action(
                self.gpu_loader.state_to_device(current_state))
            suppressed_snowball = self.suppressed_snowball(step, current_state, action)
            if suppressed_snowball:
                next_obs, r, done, _ = env.step(-1)
            else:
                next_obs, r, done, _ = env.step(action)
            if self.wandb:
                wandb.log({'Rewards/ground_truth_reward': r}, step=self.iter_count)
                if 'compass' in next_obs.keys():
                    wandb.log({'Rewards/compass': next_obs['compass']['angle']},
                              step=self.iter_count)

            replay_buffer.append_step(action, hidden, r, next_obs, done)

            if self.curriculum_training:
                curriculum_inclusion = \
                    replay_buffer.update_curriculum(step, current_max_episode_length)
                if self.wandb:
                    wandb.log({'Curriculum/inclusion_fraction': curriculum_inclusion},
                              step=self.iter_count)
            if len(replay_buffer) >= replay_buffer.replay_batch_size:
                self.train_one_batch(replay_buffer.sample(batch_size=self.batch_size))

            self.log_step()

            if time.time() > self.shutdown_time:
                print('Ending training before time cap')
                break

            if self.checkpoint_frequency > 0 and \
                    self.iter_count % self.checkpoint_frequency == 0:
                self.save_checkpoint(replay_buffer=replay_buffer)

            eval = self.eval_frequency > 0 and ((step + 1) % self.eval_frequency == 0)
            training_done = step + 1 == self.training_steps
            max_episode_length_reached = len(replay_buffer.current_trajectory()) >= \
                min(max(current_max_episode_length, self.min_training_episode_length),
                    self.max_training_episode_length)
            end_episode = done or suppressed_snowball or eval or training_done \
                or max_episode_length_reached

            training_done = step + 1 == self.training_steps
            if end_episode:
                print(f'Trajectory completed at iteration {self.iter_count}')

                self.rewards_window.append(
                    sum(replay_buffer.current_trajectory().rewards))
                self.steps_window.append(
                    len(replay_buffer.current_trajectory().rewards))
                if self.wandb:
                    wandb.log({'Rewards/train_reward': np.mean(self.rewards_window)},
                              step=self.iter_count)
                    wandb.log({'Timesteps/episodes_length': np.mean(self.steps_window)},
                              step=self.iter_count)

                if eval:
                    self.eval(env, model, replay_buffer)

                reset_env = not (training_done or suppressed_snowball)
                TrajectoryGenerator.new_trajectory(
                    env, replay_buffer,
                    reset_env=reset_env, current_obs=next_obs,
                    initial_hidden=model.initial_hidden())

            if profiler:
                profiler.step()

        print(f'{self.algorithm_name}: Training complete')
        return model, replay_buffer
