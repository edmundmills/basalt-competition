from helpers.environment import EnvironmentHelper, ObservationSpace

import math
import os
import shutil

import numpy as np


class Trajectory:
    def __init__():
        self.obs = []
        self.actions = []
        self.current_obs = None
        self.done = False
        self.number_of_frames = ObservationSpace.number_of_frames

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        done = idx + 1 == len(self) and self.done
        return(self.obs[idx], self.actions[idx], done)

    def get_current_state():
        current_step = len(self)
        frame_sequence = self.spaced_frames(current_step)
        current_pov = ObservationSpace.pov_tensor_from_single_obs(self.current_obs)
        current_inventory = ObservationSpace.dataset_obs_batch_to_inventory(
            self.current_obs)
        current_equipped = ObservationSpace.dataset_obs_batch_to_equipped(
            self.current_obs)
        return current_pov, current_inventory, current_equipped, frame_sequence

    def append_step(self, action):
        self.obs.append(current_obs)
        self.current_obs = None
        self.actions.append(action)

    def load(self, path):
        self.obs = np.load(self.trajectory_path / 'obs.npy', allow_pickle=True)
        self.actions = np.load(self.trajectory_path / 'actions.npy', allow_pickle=True)
        self.done = True

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(file=path / 'actions.npy', arr=np.array(self.actions))
        np.save(file=path / 'obs.npy', arr=np.array(self.observations))
        steps_path = path / 'steps'
        shutil.rmtree(steps_path, ignore_errors=True)
        steps_path.mkdir()
        for step in range(len(self)):
            obs, action, done = self[step]
            step_name = f'step{str(step).zfill(5)}.npy'
            step_dict = {'step': step, 'obs': obs, 'action': action, 'done': done}
            np.save(file=steps_path / step_name, arr=step_dict)

    def spaced_frames(self, current_step):
        frame_indices = [int(math.floor(current_step *
                                        frame_number / (self.number_of_frames - 1)))
                         for frame_number in range(self.number_of_frames - 1)]
        frames = np.array([self.obs[frame_idx]['pov']
                           for frame_idx in frame_indices])
        return frames


class TrajectoryGenerator:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.max_episode_length = EnvironmentHelper.max_episode_length

    def generate(self, path, save=False):
        trajectory = Trajectory()
        obs = env.reset()
        trajectory.current_obs = obs

        while not trajectory.done and len(trajectory) < self.max_episode_length:
            action = self.agent.get_action(trajectory)
            trajectory.append_step(action)
            obs, _, done, _ = env.step(action)
            trajectory.current_obs = obs
            trajectory.done = done
        if save:
            trajectory.save(path)
        return trajectory
