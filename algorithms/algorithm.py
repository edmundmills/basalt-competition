from utils.environment import ObservationSpace, ActionSpace
from utils.gpu import GPULoader
from utils.trajectories import TrajectoryGenerator

import time
import os
import wandb
from pathlib import Path
from collections import deque

import torch as th
import numpy as np


class Algorithm:
    def __init__(self, config, pretraining=False):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.gpu_loader = GPULoader(config)
        th.backends.cudnn.benchmark = True
        self.config = config
        self.max_training_episode_length = config.max_training_episode_length
        self.wandb = config.wandb
        method_config = config.pretraining if pretraining else config.method
        self.algorithm_name = method_config.name
        self.environment = config.env.name
        self.timestamps = []
        self.start_time = config.start_time
        self.training_timeout = config.training_timeout
        self.shutdown_time = self.start_time + self.training_timeout - 300
        self.update_frequency = 100
        self.eval_frequency = config.eval_frequency
        self.checkpoint_frequency = config.checkpoint_frequency
        self.name = f'{self.environment}_{self.algorithm_name}_{int(round(time.time()))}'
        self.iter_count = 1
        self.rewards_window = deque(maxlen=10)  # last N rewards
        self.steps_window = deque(maxlen=10)  # last N episode steps

    def log_step(self):
        self.iter_count += 1
        self.timestamps.append(time.time())
        self.print_update()

    def print_update(self):
        if (self.iter_count % self.update_frequency) == 0:
            print((f'{self.algorithm_name}: Iteration {self.iter_count}'
                   f' {self.iteration_rate():.2f} it/s'))

    def iteration_rate(self):
        if len(self.timestamps) < self.update_frequency - 1:
            return 0
        iterations = min(self.update_frequency, len(self.timestamps) - 1)
        duration = self.timestamps[-1] - self.timestamps[-iterations]
        if duration == 0:
            return 0
        rate = iterations / duration
        return rate

    def suppressed_snowball(self, step, current_state, action):
        if step == 0 and self.suppress_snowball_steps > 0:
            print(('Suppressing throwing snowball for'
                   f' {min(self.training_steps, self.suppress_snowball_steps)} steps'))
        elif step == self.suppress_snowball_steps and step != 0:
            print('No longer suppressing snowball')
        suppressed_snowball = step < self.suppress_snowball_steps \
            and ActionSpace.threw_snowball(current_state, action)
        if suppressed_snowball:
            print('Suppressed Snowball')
        return suppressed_snowball

    def save_checkpoint(self, replay_buffer=None, models_with_names=()):
        if replay_buffer is not None:
            if self.wandb:
                images, frame_rate = replay_buffer.recent_frames(
                    min(self.checkpoint_frequency, 1000))
                wandb.log({"video": wandb.Video(
                    images,
                    format='gif', fps=frame_rate)},
                    step=self.iter_count)

        print(f'Checkpoint saved at iteration {self.iter_count}')

    def eval(self, env, model, replay_buffer, episodes=5):
        generator = TrajectoryGenerator(env, replay_buffer)
        rewards = 0
        for i in range(episodes):
            print('Starting Evaluation Episode', i + 1)
            trajectory = generator.generate(model)
            rewards += sum(trajectory.rewards)
        print('Evaluation reward:', rewards/episodes)
        if self.wandb:
            wabdb.log({'Rewards/eval': rewards/episodes})
