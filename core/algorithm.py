from core.environment import ObservationSpace, ActionSpace
from core.gpu import GPULoader
from core.trajectories import TrajectoryGenerator

import time
import os
import wandb
from pathlib import Path
from collections import deque
import aicrowd_helper

import torch as th
import numpy as np


class Algorithm:
    def __init__(self, config, pretraining=False):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.gpu_loader = GPULoader(config)
        th.backends.cudnn.benchmark = True
        self.config = config
        self.max_training_episode_length = config.env.max_training_episode_length
        self.min_training_episode_length = config.min_training_episode_length
        self.wandb = config.wandb
        method_config = config.pretraining if pretraining else config.method
        self.algorithm_name = method_config.name
        self.environment = config.env.name
        self.timestamps = []
        self.start_time = config.start_time
        self.training_timeout = config.env.training_timeout
        self.shutdown_time = self.start_time + self.training_timeout - 300
        self.update_frequency = 100
        self.save_gifs = config.save_gifs
        self.eval_frequency = config.eval_frequency
        self.eval_episodes = config.eval_episodes
        self.checkpoint_frequency = config.checkpoint_frequency
        self.name = f'{self.environment}_{self.algorithm_name}_{int(round(time.time()))}'
        self.iter_count = 1
        self.starting_steps = method_config.starting_steps
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
            aicrowd_helper.register_progress(self.iter_count / (
                self.starting_steps + self.training_steps))

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

    def save_checkpoint(self, replay_buffer=None, model=None):
        if replay_buffer is not None and self.wandb and self.save_gifs:
            images, frame_rate = replay_buffer.recent_frames(
                min(self.checkpoint_frequency, 1000))
            wandb.log({"video": wandb.Video(
                images,
                format='gif', fps=frame_rate)},
                step=self.iter_count)
        if self.model is not None:
            model_save_path = os.path.join('train', f'{self.name}.pth')
            model.save(model_save_path)
            if self.wandb:
                model_art = wandb.Artifact("agent", type="model")
                model_art.add_file(model_save_path)
                model_art.save()

        print(f'Checkpoint saved at iteration {self.iter_count}')

    def eval(self, env, model):
        eval_path = Path('eval')
        eval_path.mkdir(exist_ok=True)
        save_path = eval_path / self.name
        generator = TrajectoryGenerator(env)
        rewards = 0
        for i in range(self.eval_episodes):
            print('Starting Evaluation Episode', i + 1)
            trajectory = generator.generate(model)
            rewards += sum(trajectory.rewards)
            trajectory.save_as_video(save_path, f'trajectory_{int(round(time.time()))}')
        print('Evaluation reward:', rewards/self.eval_episodes)
        if self.wandb:
            wandb.log({'Rewards/eval': rewards/self.eval_episodes}, step=self.iter_count)
