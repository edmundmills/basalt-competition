from core.algorithm import Algorithm
from core.datasets import ReplayBuffer, SequenceReplayBuffer
from core.trajectory_generator import TrajectoryGenerator

from collections import deque
from pathlib import Path
import time

import numpy as np
import wandb


class OnlineTraining(Algorithm):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.starting_steps = config.method.starting_steps
        self.training_steps = config.method.training_steps
        self.updates_per_step = config.method.updates_per_step

        self.batch_size = config.method.batch_size

        self.min_training_episode_length = config.env.min_training_episode_length
        self.max_training_episode_length = config.env.max_training_episode_length

        self.rewards_window = deque(maxlen=10)  # last N rewards
        self.steps_window = deque(maxlen=10)  # last N episode steps
        self.replay_buffer = self.initialize_replay_buffer(**kwargs)

    def initialize_replay_buffer(self, initial_replay_buffer=None, **kwargs):
        if initial_replay_buffer is not None:
            print((f'Using initial replay buffer'
                   f' with {len(initial_replay_buffer)} steps'))
        kwargs = dict(
            config=self.config,
            initial_replay_buffer=initial_replay_buffer
        )
        if self.config.model.lstm_layers == 0:
            replay_buffer = ReplayBuffer(**kwargs)
        else:
            replay_buffer = SequenceReplayBuffer(**kwargs)
        return replay_buffer

    def pretraining_modules(self):
        return

    def pre_train_step_modules(self, step):
        return

    def train_one_batch(self, batch):
        raise NotImplementedError

    def post_train_step_modules(self, step):
        return

    def training_step(self):
        training_metrics = None
        for i in range(self.updates_per_step):
            batch = self.replay_buffer.sample(batch_size=self.batch_size)
            metrics = self.train_one_batch(batch)

            # collect and log metrics:
            if training_metrics is None:
                training_metrics = {}
                for k, v in iter(metrics.items()):
                    training_metrics[k] = [v]
            else:
                for k, v in iter(metrics.items()):
                    training_metrics[k].append(v)
        for k, v in iter(training_metrics.items()):
            training_metrics[k] = sum(v) / len(v)
        return training_metrics

    def max_episode_length(self, step):
        if not (self.curriculum_scheduler and
                self.curriculum_scheduler.variable_training_episode_length):
            return self.max_training_episode_length
        else:
            return self.curriculum_scheduler.max_episode_length(self, step)

    def conditionally_increment_episode(self, step, current_trajectory):
        eval = self.eval_frequency > 0 and ((step + 1) % self.eval_frequency == 0)
        training_done = self.training_done(step)
        max_episode_length_reached = \
            len(current_trajectory) >= self.max_episode_length(step)

        end_episode = current_trajectory.done or eval or training_done \
            or max_episode_length_reached or current_trajectory.suppressed_termination()

        if end_episode:
            print(f'Trajectory completed at iteration {self.iter_count}')
            self.rewards_window.append(sum(current_trajectory.rewards))
            self.steps_window.append(len(current_trajectory.rewards))
            if self.wandb:
                wandb.log({'Rewards/train_reward': np.mean(self.rewards_window),
                           'Timesteps/episodes_length': np.mean(self.steps_window)},
                          step=self.iter_count)

            if eval:
                self.eval()

            reset_env = not (training_done or current_trajectory.suppressed_termination())
            self.trajectory_generator.start_new_trajectory(reset_env=reset_env)

    def eval(self):
        eval_path = Path('eval')
        eval_path.mkdir(exist_ok=True)
        save_path = eval_path / self.name
        rewards = 0
        for i in range(self.eval_episodes):
            print('Starting Evaluation Episode', i + 1)
            trajectory = self.trajectory_generator.generate()
            rewards += sum(trajectory.rewards)
            trajectory.save_as_video(save_path, f'trajectory_{int(round(time.time()))}')
        print('Evaluation reward:', rewards/self.eval_episodes)
        if self.wandb:
            wandb.log({'Rewards/eval': rewards/self.eval_episodes}, step=self.iter_count)

    def __call__(self, env, profiler=None):
        self.pretraining_modules()

        print((f'{self.algorithm_name}: training for {self.training_steps}'))

        self.trajectory_generator = TrajectoryGenerator(env, self.agent,
                                                        self.config, self.replay_buffer,
                                                        training=True)
        self.trajectory_generator.start_new_trajectory()

        for step in range(self.training_steps):

            action_metrics = self.trajectory_generator.env_interaction_step(step)

            pretrain_metrics = self.pre_train_step_modules(step)

            # train models
            if len(self.replay_buffer) > self.batch_size:
                training_metrics = self.training_step()
            else:
                training_metrics = {}

            posttrain_metrics = self.post_train_step_modules(step)

            metrics = {**action_metrics, **pretrain_metrics,
                       **training_metrics, **posttrain_metrics}

            self.increment_step(metrics, profiler)

            if self.shutdown_time_reached():
                break

            self.save_checkpoint(replay_buffer=self.replay_buffer, model=self.agent)

            self.conditionally_increment_episode(step,
                                                 self.replay_buffer.current_trajectory())

        print(f'{self.algorithm_name}: Training complete')
        return self.agent, self.replay_buffer
