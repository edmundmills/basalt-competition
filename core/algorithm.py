import aicrowd_helper
from core.gpu import GPULoader
from core.trajectories import TrajectoryGenerator

from collections import deque
import os
from pathlib import Path
import time

import numpy as np
import torch as th
import wandb


class Algorithm:
    def __init__(self, config, pretraining=False):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        th.backends.cudnn.benchmark = True
        self.gpu_loader = GPULoader(config)
        self.config = config
        self.wandb = config.wandb
        self.environment = config.env.name
        self.algorithm_name = config.method.name
        self.name = f'{self.environment}_{self.algorithm_name}_{int(round(time.time()))}'

        # context
        if config.context.name == 'MineRL':
            self.context = MineRLContext(config)

        self.start_time = config.start_time
        self.training_timeout = config.training_timeout
        self.shutdown_time = self.start_time + self.training_timeout - 300

        self.update_frequency = 100
        self.checkpoint_frequency = config.checkpoint_frequency
        self.eval_frequency = config.eval_frequency
        self.eval_episodes = config.eval_episodes
        self.save_gifs = config.save_gifs

        self.starting_steps = config.method.starting_steps
        self.min_training_episode_length = config.env.min_training_episode_length
        self.max_training_episode_length = config.env.max_training_episode_length

        self.timestamps = []
        self.iter_count = 1
        self.rewards_window = deque(maxlen=10)  # last N rewards
        self.steps_window = deque(maxlen=10)  # last N episode steps

    def log_step(self, metrics, profiler):
        self.iter_count += 1
        self.timestamps.append(time.time())
        self.print_update()

        if self.wandb:
            wandb.log(metrics, step=self.iter_count)

        if profiler:
            profiler.step()

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

    def max_episode_length(self, step):
        if not (self.curriculum_scheduler and
                self.curriculum_scheduler.variable_training_episode_length):
            return self.max_training_episode_length
        else:
            return self.curriculum_scheduler.max_episode_length(self, step)

    def training_done(self, step):
        return step + 1 == self.training_steps

    def shutdown_time_reached(self):
        if time.time() > self.shutdown_time:
            print('Ending training before time cap')
            return True
        else:
            return False

    def new_episode_if_time(self, step, current_trajectory, suppressed_snowball=False):
        eval = self.eval_frequency > 0 and ((step + 1) % self.eval_frequency == 0)
        training_done = self.training_done(step)
        max_episode_length_reached = \
            len(current_trajectory) >= self.max_episode_length(step)
        end_episode = done or suppressed_snowball or eval or training_done \
            or max_episode_length_reached

        if end_episode:
            print(f'Trajectory completed at iteration {self.iter_count}')
            self.rewards_window.append(sum(current_trajectory.rewards))
            self.steps_window.append(len(current_trajectory.rewards))
            if self.wandb:
                wandb.log({'Rewards/train_reward': np.mean(self.rewards_window),
                           'Timesteps/episodes_length': np.mean(self.steps_window)},
                          step=self.iter_count)

            if eval:
                self.eval(env, model)

            reset_env = not (training_done or suppressed_snowball)
            self.trajectory_generator.new_trajectory(
                reset_env=reset_env, current_state=current_trajectory.current_state())
