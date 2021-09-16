from algorithms.loss_functions.iqlearn import IQLearn
from algorithms.loss_functions.sqil import Sqil
from helpers.environment import ObservationSpace, ActionSpace
from helpers.datasets import MixedReplayBuffer

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

import wandb
import os


class OnlineImitation:
    def __init__(self, loss_function_name, termination_critic=None):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.termination_critic = termination_critic
        self.loss_function_name = loss_function_name

    def __call__(self, model, env, expert_dataset, run, profiler=None):
        if self.loss_function_name == 'sqil':
            self.loss_function = Sqil(model, run)
        elif self.loss_function_name == 'iqlearn':
            self.loss_function = IQLearn(model, run)

        optimizer = th.optim.Adam(model.parameters(),
                                  lr=run.config['learning_rate'])

        replay_buffer = MixedReplayBuffer(expert_dataset=expert_dataset,
                                          batch_size=run.config['batch_size'],
                                          expert_sample_fraction=0.5,
                                          n_observation_frames=model.n_observation_frames)

        obs = env.reset()
        replay_buffer.current_trajectory().append_obs(obs)

        for step in range(run.config['training_steps']):
            iter_count = step + 1

            current_state = replay_buffer.current_trajectory().current_state(
                n_observation_frames=model.n_observation_frames)
            action = model.get_action(current_state)
            if ActionSpace.threw_snowball(current_state, action):
                print(f'Threw Snowball at step {iter_count}')
                if self.termination_critic is not None:
                    reward = self.termination_critic.termination_reward(current_state)
                    print(f'Termination reward: {reward:.2f}')
                    if run.wandb:
                        wandb.log({'termination_reward': reward})
                else:
                    reward = 0
            else:
                reward = 0

            next_obs, _, done, _ = env.step(action)
            replay_buffer.append_step(action, reward, next_obs, done)

            if len(replay_buffer) >= replay_buffer.replay_batch_size:
                loss = self.loss_function(replay_buffer.sample_expert(),
                                          replay_buffer.sample_replay())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if run.wandb:
                    wandb.log({'loss': loss.detach()})
                run.step()

            run.print_update()

            if done:
                print(f'Trajectory completed at step {iter_count}')
                replay_buffer.new_trajectory()
                obs = env.reset()
                replay_buffer.current_trajectory().append_obs(obs)

            if profiler:
                profiler.step()
            if run.checkpoint_freqency and iter_count % run.checkpoint_freqency == 0 \
                    and iter_count < run.config['training_steps']:
                model.save(os.path.join('train', f'{run.name}.pth'))
                print(f'Checkpoint saved at step {iter_count}')

        print('Training complete')
