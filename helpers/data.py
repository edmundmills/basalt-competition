from helpers.trajectories import Trajectory

from pathlib import Path
import os
from zipfile import ZipFile
import time
import shutil


import numpy as np

import minerl


def pre_process_expert_trajectories():
    data_dir = os.getenv('MINERL_DATA_ROOT')
    data_root = Path(data_dir)
    for environment_path in data_root.iterdir():
        environment_name = environment_path.name
        if environment_name not in ['MineRLBasaltBuildVillageHouse-v0',
                                    'MineRLBasaltCreateVillageAnimalPen-v0',
                                    'MineRLBasaltFindCave-v0',
                                    'MineRLBasaltMakeWaterfall-v0']:
            continue

        # Move nested data dir
        if environment_name == 'MineRLBasaltCreateVillageAnimalPen-v0':
            sub_dir = environment_path / 'MineRLBasaltCreateAnimalPenPlains-v0'
            if sub_dir.is_dir():
                for trajectory_path in sub_dir.iterdir():
                    if trajectory_path.is_dir():
                        target = sub_dir.parent / trajectory_path.name
                        trajectory_path.replace(target)
                sub_dir.rmdir()

        data = minerl.data.make(environment_name,  data_dir=data_dir, num_workers=4)
        trajectory_paths = environment_path.iterdir()
        # trajectory_paths = [environment_path /
        #                     'v3_accomplished_pattypan_squash_ghost-6_765-1145']
        for trajectory_path in trajectory_paths:
            if not trajectory_path.is_dir():
                continue

            trajectory = Trajectory()
            for obs, action, _, _, done in data.load_data(str(trajectory_path)):
                trajectory.obs.append(obs)
                trajectory.actions.append(action)
                trajectory.done = done
            trajectory.save(trajectory_path)


def zip_demonstrations(paths, filename):
    parent_dir = paths[0].parent
    zip_dir = parent_dir / f'zip_temp_{time.time()}'
    zip_dir.mkdir(exist_ok=True)
    for path in paths:
        target = zip_dir / path.name
        path.replace(target)
    with ZipFile(parent_dir / filename, 'w') as zip:
        zip.write(zip_dir)
    shutil.rmtree(zip_dir, ignore_errors=True)
    print('Files Zipped')


if __name__ == "__main__":
    os.environ['MINERL_DATA_ROOT'] = '../../code/basalt/data'
    convert_data(data_dir)
