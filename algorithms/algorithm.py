from helpers.environment import ObservationSpace, ActionSpace

import time
import os
import wandb
from pathlib import Path

import torch as th


class Algorithm:
    def __init__(self, config):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.config = config
        self.wandb = config.wandb
        self.algorithm_name = config.algorithm
        self.environment = config.env.name
        self.timestamps = []
        self.update_frequency = 50
        self.checkpoint_freqency = config.checkpoint_freqency
        self.name = f'{self.environment}_{self.algorithm_name}_{int(round(time.time()))}'
        save_path = Path('training_runs') / self.name
        os.makedirs(save_path.as_posix(), exist_ok=True)
        self.save_path = save_path

    def generate_random_trajectories(self, replay_buffer, env, steps):
        current_state = self.start_new_trajectory(env, replay_buffer)

        # generate random trajectories
        for step in range(steps):
            iter_count = step + 1
            action = ActionSpace.random_action()
            replay_buffer.current_trajectory().actions.append(action)

            suppressed_snowball = ActionSpace.threw_snowball(current_state, action)
            if suppressed_snowball:
                print('Snowball suppressed')
                obs, _, done, _ = env.step(-1)
            else:
                obs, _, done, _ = env.step(action)

            replay_buffer.current_trajectory().append_obs(obs)
            replay_buffer.current_trajectory().done = done
            next_state = replay_buffer.current_state()

            reward = 0
            replay_buffer.current_trajectory().rewards.append(reward)

            replay_buffer.increment_step()
            current_state = next_state

            if done or (iter_count % 1000 == 0 and iter_count != steps):
                print(f'Starting trajectory completed at step {iter_count}')
                current_state = self.start_new_trajectory(env, replay_buffer)
            elif suppressed_snowball:
                replay_buffer.current_trajectory().done = True
                replay_buffer.new_trajectory()
                replay_buffer.current_trajectory().append_obs(obs)
                current_state = replay_buffer.current_state()

            self.log_step()

    def start_new_trajectory(self, env, replay_buffer):
        if len(replay_buffer.current_trajectory()) > 0:
            replay_buffer.new_trajectory()
        obs = env.reset()
        replay_buffer.current_trajectory().append_obs(obs)
        current_state = replay_buffer.current_state()
        return current_state

    def log_step(self):
        self.timestamps.append(time.time())
        self.print_update()

    def print_update(self):
        iter_count = len(self.timestamps)
        if (iter_count % self.update_frequency) == 0 and len(self.timestamps) > 2:
            print(f'Iteration {iter_count} {self.iteration_rate():.2f} it/s')

    def iteration_rate(self):
        iterations = min(self.update_frequency, len(self.timestamps) - 1)
        duration = self.timestamps[-1] - self.timestamps[-iterations]
        rate = iterations / duration
        return rate

    def save_checkpoint(self, iter_count, replay_buffer=None, models_with_names=()):
        for model, name in models_with_names:
            model.save(os.path.join('train', f'{name}.pth'))
        if replay_buffer is not None:
            gif_paths = []
            gif_name = f'step_{iter_count}_tr_{len(replay_buffer.trajectories)}'
            gif_path = replay_buffer.current_trajectory().save_gif(self.save_path,
                                                                   gif_name)
            gif_paths.append(gif_path)
            if len(replay_buffer.trajectories) > 1:
                gif_name2 = f'step_{iter_count}_tr_{len(replay_buffer.trajectories)-1}'
                previous_trajectory = replay_buffer.trajectories[-2]
                gif_path2 = previous_trajectory.save_gif(self.save_path, gif_name2)
                gif_paths.append(gif_path2)
            if self.wandb:
                gif_art = wandb.Artifact("checkpoint", type="gif")
                for gif_path in gif_paths:
                    gif_art.add_file(gif_path)
                gif_art.save()

        print(f'Checkpoint saved at step {iter_count}')
