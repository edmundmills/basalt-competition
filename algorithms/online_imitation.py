from algorithms.algorithm import Algorithm
from algorithms.loss_functions.iqlearn import IQLearnLoss
from algorithms.loss_functions.sqil import SqilLoss
from helpers.environment import ObservationSpace, ActionSpace
from helpers.datasets import MixedReplayBuffer

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

import wandb
import os


class OnlineImitation(Algorithm):
    def __init__(self, expert_dataset, model, config, termination_critic=None,
                 initial_replay_buffer=None, initial_iter_count=0):
        super().__init__(config)
        self.termination_critic = termination_critic
        self.lr = config['learning_rate']
        self.starting_steps = config['starting_steps']
        self.training_steps = config['training_steps']
        self.batch_size = config['batch_size']
        self.model = model
        self.expert_dataset = expert_dataset
        self.initial_replay_buffer = initial_replay_buffer
        self.iter_count += initial_iter_count

    def __call__(self, env, profiler=None):
        model = self.model
        expert_dataset = self.expert_dataset
        initial_replay_buffer = self.initial_replay_buffer

        if self.config['loss_function'] == 'sqil':
            self.loss_function = SqilLoss(model, self.config)
        elif self.config['loss_function'] == 'iqlearn':
            self.loss_function = IQLearnLoss(model, self.config)

        optimizer = th.optim.Adam(model.parameters(),
                                  lr=self.lr)

        replay_buffer = MixedReplayBuffer(expert_dataset=expert_dataset,
                                          batch_size=self.batch_size,
                                          expert_sample_fraction=0.5,
                                          n_observation_frames=model.n_observation_frames,
                                          initial_replay_buffer=initial_replay_buffer)

        if self.starting_steps > 0 and initial_replay_buffer is None:
            self.generate_random_trajectories(replay_buffer, env, self.starting_steps)

        self.start_new_trajectory(env, replay_buffer)

        print((f'{self.algorithm_name}: Starting training'
               f' for {self.training_steps} steps (iteration {self.iter_count})'))

        for step in range(self.training_steps):
            current_state = replay_buffer.current_trajectory().current_state(
                n_observation_frames=model.n_observation_frames)
            action = model.get_action(current_state)
            if ActionSpace.threw_snowball(current_state, action):
                print(f'Threw Snowball at iteration {self.iter_count}')
                if self.termination_critic is not None:
                    reward = self.termination_critic.termination_reward(current_state)
                    print(f'Termination reward: {reward:.2f}')
                    if self.wandb:
                        wandb.log({'termination_reward': reward}, step=self.iter_count)
                else:
                    reward = 0
            else:
                reward = 0

            next_obs, _, done, _ = env.step(action)
            replay_buffer.append_step(action, reward, next_obs, done)

            if len(replay_buffer) >= replay_buffer.replay_batch_size:
                loss, metrics = self.loss_function(replay_buffer.sample_expert(),
                                                   replay_buffer.sample_replay())

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

            if done:
                print(f'Trajectory completed at iteration {self.iter_count}')
                self.start_new_trajectory(env, replay_buffer)

            if profiler:
                profiler.step()

        print(f'{self.algorithm_name}: Training complete')
        return model, replay_buffer
