from helpers.environment import ObservationSpace, ActionSpace

import time
from pathlib import Path
import torch as th


class Algorithm:
    def __init__(self, config):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.config = config
        self.wandb = config['wandb']
        self.algorithm_name = config['algorithm']
        self.environment = config['environment']
        self.timestamps = []
        self.update_frequency = 50
        self.checkpoint_freqency = 1000
        self.name = f'{self.environment}_{self.algorithm_name}_{int(round(time.time()))}'
        save_path = Path('training_runs') / self.name
        save_path.mkdir(exist_ok=True)
        self.save_path = save_path

    def generate_random_trajectories(self, replay_buffer, env, steps):
        current_state = self.start_new_trajectory(env, replay_buffer)

        # generate random trajectories
        for step in range(steps):
            iter_count = step + 1
            action = ActionSpace.random_action()
            replay_buffer.current_trajectory().actions.append(action)

            if ActionSpace.threw_snowball(current_state, action):
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
