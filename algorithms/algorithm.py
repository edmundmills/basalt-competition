from helpers.environment import ObservationSpace, ActionSpace

import time
import os
import wandb
from pathlib import Path

import torch as th
import numpy as np


class Algorithm:
    def __init__(self, config, pretraining=False):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
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
        # # save_path = Path('train') / self.name
        # # os.makedirs(save_path.as_posix(), exist_ok=True)
        # self.save_path = save_path
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
        # for model, name in models_with_names:
        #     model.save(os.path.join(self.save_path, f'{name}.pth'))
        if replay_buffer is not None:
            # video_paths = []
            # video_name = f'trajectory_{len(replay_buffer.trajectories)}'
            # video_path = replay_buffer.current_trajectory().save_as_video(
            #           self.save_path, video_name)
            # video_paths.append(video_path)
            # if len(replay_buffer.trajectories) > 1:
            #     video_name2 = f'trajectory_{len(replay_buffer.trajectories)-1}'
            #     previous_trajectory = replay_buffer.trajectories[-2]
            #     video_path2 = previous_trajectory.save_as_video(self.save_path,
            #                                                     video_name2)
            #     video_paths.append(video_path2)
            if self.wandb:
                images, frame_rate = replay_buffer.recent_frames(
                    min(self.checkpoint_frequency, 1000))
                wandb.log({"video": wandb.Video(
                    images,
                    format='gif', fps=frame_rate)},
                    step=self.iter_count)

        print(f'Checkpoint saved at iteration {self.iter_count}')
