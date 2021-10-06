from utils.environment import ObservationSpace, ActionSpace
from utils.gpu import GPULoader

import time
import os
import wandb
from pathlib import Path

import torch as th
import numpy as np


class Algorithm:
    def __init__(self, config, pretraining=False):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.gpu_loader = GPULoader()
        th.backends.cudnn.benchmark = True
        self.config = config
        self.wandb = config.wandb
        method_config = config.pretraining if pretraining else config.method
        self.algorithm_name = method_config.name
        self.environment = config.env.name
        self.timestamps = []
        self.update_frequency = 100
        self.checkpoint_frequency = config.checkpoint_frequency
        self.name = f'{self.environment}_{self.algorithm_name}_{int(round(time.time()))}'
        self.iter_count = 1

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
