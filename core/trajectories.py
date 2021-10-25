from contexts.minerl.environment import MineRLContext, SnowballHelper
from core.gpu import GPULoader
from core.state import State, Transition, Sequence

import math
from pathlib import Path
from collections import OrderedDict, deque

import numpy as np
import torch as th

import cv2


class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.done = False
        self.additional_step_data = {}

    def __len__(self):
        return max(0, len(self.states) - 1)

    def __getitem__(self, idx):
        is_last_step = idx + 1 == len(self)
        done = is_last_step and self.done
        next_state = self.states[idx + 1]
        reward = self.rewards[idx]
        return Transition(self.states[idx], self.actions[idx], reward, next_state, done)

    def current_state(self):
        return self.states[-1]

    def update_hidden(self, idx, new_hidden):
        is_last_step = idx + 1 == len(self)
        done = is_last_step and self.done
        if not done:
            idx += 1
        next_state = self.states[idx]
        *state_components, hidden = next_state
        self.states[idx] = State(*state_components, new_hidden)

    def get_sequence(self, last_step_idx, sequence_length):
        is_last_step = last_step_idx + 1 == len(self)
        dones = th.zeros(sequence_length)
        dones[-1] = 1 if is_last_step and self.done else 0
        actions = th.LongTensor(
            self.actions[last_step_idx - sequence_length:last_step_idx])
        states = self.states[last_step_idx - sequence_length:last_step_idx + 1]
        states = State(*[th.stack(state_component) for state_component in zip(*states)])
        rewards = th.FloatTensor(
            self.rewards[last_step_idx - sequence_length:last_step_idx])
        return Sequence(states, actions, rewards, dones)


class TrajectoryGenerator:
    def __init__(self, env, actor, config, replay_buffer=None):
        self.env = env
        self.actor = actor
        self.replay_buffer = replay_buffer
        self.config = config
        self.gpu_loader = GPULoader(config)
        if config.context.name == 'MineRL':
            self.context = MineRLContext(config)
            self.snowball_helper = self.context.snowball_helper

    def generate(self, max_episode_length=100000, print_actions=False):
        trajectory = Trajectory()
        state = self.env.reset()

        while not trajectory.done and len(trajectory) < max_episode_length:
            trajectory.states.append(state)
            current_state = trajectory.current_state()
            action, hidden = model.get_action(gpu_loader.state_to_device(current_state),
                                              iter_count=len(trajectory))
            trajectory.actions.append(action)
            state, reward, done, _ = self.env.step(action)
            # if print_actions:
            #     print(action, ActionSpace.action_name(action),
            #           f'(equipped: {ActionSpace.equipped_item(current_state)})')
            trajectory.rewards.append(reward)
            trajectory.done = done
            if self.replay_buffer:
                self.replay_buffer.increment_step()
        print('Finished generating trajectory'
              f' (reward: {sum(trajectory.rewards)}, length: {len(trajectory.rewards)})')
        return trajectory

    def new_trajectory(env, replay_buffer, reset_env=True,
                       current_state=None):
        if len(replay_buffer.current_trajectory()) > 0:
            replay_buffer.new_trajectory()
        state = env.reset() if reset_env else current_state
        replay_buffer.current_trajectory().states.append(state)
        return replay_buffer.current_state()

    def start_new_trajectory(self, **kwargs):
        current_state = TrajectoryGenerator.new_trajectory(
            self.env, self.replay_buffer, **kwargs)
        return current_state

    def random_action(self):
        action = np.random.choice(self.context.actions)
        return action

    def random_trajectories(self, steps, max_length=1000):
        print(f'Generating random trajectories for {steps} steps')
        current_state = self.start_new_trajectory()

        # generate random trajectories
        for step in range(steps):
            action = self.random_action()
            suppressed_snowball = self.snowball_helper.threw_snowball(current_state,
                                                                      action)
            if suppressed_snowball:
                next_state, _reward, done, _ = self.env.step(-1)
                reward = -1
            else:
                next_state, reward, done, _ = self.env.step(action)

            self.replay_buffer.append_step(action, reward, next_state, done)

            current_state = self.replay_buffer.current_state()

            if done or (step % max_length == 0 and step != steps):
                print(f'Random trajectory completed at step {step}')
                current_state = self.start_new_trajectory()
            elif suppressed_snowball:
                current_state = self.start_new_trajectory(reset_env=False,
                                                          current_state=state)

        trajectory_count = len(self.replay_buffer.trajectories)
        print(f'Finished generating {trajectory_count} random trajectories')
        return self.replay_buffer

        def env_interaction_step(self):
            metrics = {}
            current_state = self.replay_buffer.current_state()
            action, hidden = self.actor.get_action(
                self.gpu_loader.state_to_device(current_state))

            suppressed_snowball = self.snowball_helper.suppressed_snowball(
                step, current_state, action) if self.snowball_helper else False

            if suppressed_snowball:
                next_state, reward, done, _ = self.env.step(-1)
            else:
                next_state, reward, done, _ = self.env.step(action)

            update_hidden(next_state, hidden)
            metrics['Rewards/ground_truth_reward'] = reward
            self.replay_buffer.append_step(action, reward, next_state, done,
                                           suppressed_snowball=suppressed_snowball)
            return metrics
