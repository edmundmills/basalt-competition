from utils.environment import ObservationSpace, ActionSpace
from utils.gpu import GPULoader

import math
from pathlib import Path
from collections import OrderedDict, deque

import numpy as np
import torch as th

import cv2


class Trajectory:
    def __init__(self, n_observation_frames):
        self.n_observation_frames = n_observation_frames
        self.framestack = deque(maxlen=self.n_observation_frames)
        self.states = []
        self.actions = []
        self.rewards = []
        self.additional_data = OrderedDict()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        is_last_step = idx + 1 == len(self)
        done = is_last_step and self.done
        if done:
            next_state = self.states[idx]
        elif is_last_step:
            next_state = None
        else:
            next_state = self.states[idx + 1]
        reward = self.rewards[idx] if len(self.rewards) > 0 else 0
        return self.states[idx], self.actions[idx], next_state, done, reward

    def append_obs(self, obs, hidden=None):
        pov = ObservationSpace.obs_to_pov(obs)
        # initialize framestack
        while len(self.framestack) < self.n_observation_frames:
            self.framestack.append(pov)
        self.framestack.append(pov)
        pov = th.cat(list(self.framestack), dim=0)
        items = ObservationSpace.obs_to_items(obs)
        if hidden is not None:
            state = pov, items, hidden
        else:
            state = pov, items
        self.states.append(state)

    def current_state(self, **kwargs):
        current_idx = len(self) - 1
        state = self.states[current_idx]
        return state

    def get_pov(self, idx):
        pov = self.states[idx][0]
        single_frame = pov[-3:, :, :]
        return single_frame

    def get_segment(self, last_step_idx, segment_length):
        is_last_step = last_step_idx + 1 == len(self)
        done = 1 if is_last_step and self.done else 0
        actions = self.actions[last_step_idx + 1 - segment_length, last_step_idx + 1]
        states = self.states[last_step_idx + 1 - segment_length, last_step_idx + 2]
        states = [th.stack(state_component) for state_component in zip(*states)]
        _next_states = ([], [], [])
        _rewards = []
        return states, actions, _next_states, done, _rewards

    def save_as_video(self, save_dir_path, filename):
        save_dir_path = Path(save_dir_path)
        save_dir_path.mkdir(exist_ok=True)
        images, frame_rate = self.as_video_frames()
        video_path = save_dir_path / f'{filename}.mp4'
        frame_size = (64, 64)
        out = cv2.VideoWriter(str(video_path),
                              cv2.VideoWriter_fourcc(*'FMP4'),
                              frame_rate, frame_size)
        for img in images:
            out.write(img)
        out.release()
        return video_path

    def as_video_frames(self):
        total_steps = len(self)
        frame_skip = 2
        frames = min(int(round(total_steps / (frame_skip + 1))), total_steps)
        step_rate = 20  # steps / second
        frame_rate = int(round(step_rate / (frame_skip + 1)))
        duration = frames / frame_rate
        step_indices = [frame * (frame_skip + 1) for frame in range(frames)]
        images = [(self.get_pov(idx).numpy()).astype(
            np.uint8).transpose(1, 2, 0)[..., ::-1]
            for idx in step_indices]
        return images, frame_rate


class TrajectoryGenerator:
    def __init__(self, env, replay_buffer=None, lstm_hidden_size=0):
        self.env = env
        self.replay_buffer = replay_buffer
        self.gpu_loader = GPULoader()
        self.lstm_hidden_size = lstm_hidden_size
        self.initial_hidden = th.zeros(self.lstm_hidden_size*2) \
            if lstm_hidden_size > 0 else None

    def generate(self, model, max_episode_length=100000):
        trajectory = Trajectory(n_observation_frames=model.n_observation_frames)
        obs = self.env.reset()
        hidden = model.lstm.initial_hidden if model.lstm else None

        while not trajectory.done and len(trajectory) < max_episode_length:
            trajectory.append_obs(obs, hidden)
            state = trajectory.current_state()
            action, hidden = model.get_action(
                self.gpu_loader.state_to_device(current_state))
            trajectory.actions.append(action)
            obs, _, done, _ = self.env.step(action)
            trajectory.done = done
        print('Finished generating trajectory')
        return trajectory

    def new_trajectory(env, replay_buffer, reset_env=True,
                       current_obs=None, initial_hidden=None):
        if len(replay_buffer.current_trajectory()) > 0:
            replay_buffer.new_trajectory()
        obs = env.reset() if reset_env else current_obs
        replay_buffer.current_trajectory().append_obs(obs, initial_hidden)
        return replay_buffer.current_state()

    def start_new_trajectory(self, **kwargs):
        current_state = TrajectoryGenerator.new_trajectory(
            self.env, self.replay_buffer, initial_hidden=self.initial_hidden, **kwargs)
        return current_state

    def random_trajectories(self, steps):
        print(f'Generating random trajectories for {steps} steps')

        current_state = self.start_new_trajectory()

        # generate random trajectories
        for step in range(steps):
            action = ActionSpace.random_action()
            self.replay_buffer.current_trajectory().actions.append(action)

            suppressed_snowball = ActionSpace.threw_snowball(current_state, action)
            if suppressed_snowball:
                obs, _, done, _ = self.env.step(-1)
                reward = -1
            else:
                obs, _, done, _ = self.env.step(action)
                reward = 0

            self.replay_buffer.current_trajectory().append_obs(obs, self.initial_hidden)
            self.replay_buffer.current_trajectory().done = done
            next_state = self.replay_buffer.current_state()

            self.replay_buffer.current_trajectory().rewards.append(reward)

            self.replay_buffer.increment_step()
            current_state = next_state

            if done or (step % 1000 == 0 and step != steps):
                print(f'Random trajectory completed at step {step}')
                current_state = self.start_new_trajectory()
            elif suppressed_snowball:
                current_state = self.start_new_trajectory(reset_env=False,
                                                          current_obs=obs)

        trajectory_count = len(self.replay_buffer.trajectories)
        print(f'Finished generating {trajectory_count} random trajectories')
        return self.replay_buffer
