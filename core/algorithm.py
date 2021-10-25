import aicrowd_helper
from core.data_augmentation import DataAugmentation
from core.environment import create_context
from core.gpu import GPULoader

from pathlib import Path
import time

import numpy as np
import torch as th
import wandb


class Algorithm:
    def __init__(self, config, initial_iter_count=0, **kwargs):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        th.backends.cudnn.benchmark = True
        self.gpu_loader = GPULoader(config)
        self.config = config
        self.wandb = config.wandb
        self.environment = config.env.name
        self.algorithm_name = config.method.name
        self.name = f'{self.environment}_{self.algorithm_name}_{int(round(time.time()))}'
        self.context = create_context(config)
        self.augmentation = DataAugmentation(config)

        self.start_time = config.start_time
        self.training_timeout = config.training_timeout
        self.shutdown_time = self.start_time + self.training_timeout - 300

        self.logging_frequency = 100
        self.checkpoint_frequency = config.checkpoint_frequency
        self.eval_frequency = config.eval_frequency
        self.eval_episodes = config.eval_episodes
        self.save_gifs = config.save_gifs

        self.timestamps = []
        self.iter_count = 1 + initial_iter_count

    def increment_step(self, metrics, profiler):
        self.iter_count += 1
        self.timestamps.append(time.time())
        self.print_update()

        if self.wandb:
            wandb.log(metrics, step=self.iter_count)

        if profiler:
            profiler.step()

    def print_update(self):
        if (self.iter_count % self.logging_frequency) == 0:
            print((f'{self.algorithm_name}: Iteration {self.iter_count}'
                   f' {self.iteration_rate():.2f} it/s'))
            aicrowd_helper.register_progress(self.iter_count / (
                self.starting_steps + self.training_steps))

    def iteration_rate(self):
        if len(self.timestamps) < self.logging_frequency - 1:
            return 0
        iterations = min(self.logging_frequency, len(self.timestamps) - 1)
        duration = self.timestamps[-1] - self.timestamps[-iterations]
        if duration == 0:
            return 0
        rate = iterations / duration
        return rate

    def save_checkpoint(self, replay_buffer=None, model=None):
        if not (self.checkpoint_frequency > 0 and
                self.iter_count % self.checkpoint_frequency == 0):
            return

        if replay_buffer is not None and self.wandb and self.save_gifs:
            images, frame_rate = replay_buffer.recent_frames(
                min(self.checkpoint_frequency, 1000))
            wandb.log({"video": wandb.Video(
                images,
                format='gif', fps=frame_rate)},
                step=self.iter_count)
        if self.model is not None:
            model_save_path = Path('train') / f'{self.name}.pth'
            model.save(model_save_path)
            if self.wandb:
                model_art = wandb.Artifact("agent", type="model")
                model_art.add_file(model_save_path)
                model_art.save()

        print(f'Checkpoint saved at iteration {self.iter_count}')

    def training_done(self, step):
        return step + 1 == self.training_steps

    def shutdown_time_reached(self):
        if time.time() > self.shutdown_time:
            print('Ending training before time cap')
            return True
        else:
            return False
