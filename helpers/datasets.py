from helpers.environment import ObservationSpace

from pathlib import Path
import os

import torch as th
import math
import numpy as np

from torch.utils.data import Dataset


class StepDataset(Dataset):
    def __init__(self,
                 data_root=os.getenv('MINERL_DATA_ROOT'),
                 environments=[]):
        self.environments = environments if environments is not [] else [
            os.getenv('MINERL_ENVIRONMENT')]
        self.data_root = Path(data_root)
        step_paths = []
        for environment_name in environments:
            environment_path = self.data_root / environment_name
            step_paths.extend(self._get_step_data(environment_path))
        self.step_paths = step_paths

    def _get_step_data(self, environment_path):
        step_paths = []
        for trajectory_path in sorted(environment_path.iterdir()):
            steps_dir_path = trajectory_path / 'steps'
            if not steps_dir_path.is_dir():
                continue
            trajecory_step_paths = sorted(steps_dir_path.iterdir())
            step_paths.extend(trajecory_step_paths)
        return step_paths

    def __len__(self):
        return len(self.step_paths)

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()
        if idx + 1 > len(self):
            raise IndexError('list index out of range')
        step_path = self.step_paths[idx]
        step_dict = np.load(step_path, allow_pickle=True).item()
        return step_dict['obs'], step_dict['action'], step_dict['done']


class MultiFrameDataset(StepDataset):
    def __init__(self,
                 data_root=os.getenv('MINERL_DATA_ROOT'),
                 environments=[]):
        super().__init__(data_root, environments)
        self.number_of_frames = ObservationSpace.number_of_frames

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()
        if idx + 1 > len(self):
            raise IndexError('list index out of range')
        step_path = self.step_paths[idx]
        step_dict = np.load(step_path, allow_pickle=True).item()
        step_number = step_dict['step']
        frame_indices = idx - step_number + self.frame_indices(step_number)
        frame_sequence = np.array([np.load(step_path, allow_pickle=True).item()[
            'obs']['pov'] for step_idx in step_idxes])
        step_dict['obs']['frame_sequence'] = frame_sequence
        return step_dict['obs'], step_dict['action'], step_dict['done']

    def frame_indices(self, step_number):
        return [int(math.floor(step_number *
                               frame_number / (self.number_of_frames - 1)))
                for frame_number in range(self.number_of_frames - 1)]
