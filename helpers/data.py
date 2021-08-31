from pathlib import Path
import shutil

import numpy as np

import minerl


def convert_data(data_dir):
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
        # trajectory_paths = environment_path.iterdir()
        trajectory_paths = [environment_path /
                            'v3_accomplished_pattypan_squash_ghost-6_765-1145']
        for trajectory_path in trajectory_paths:
            if not trajectory_path.is_dir():
                continue

            step = 0
            steps_path = trajectory_path / 'steps'
            shutil.rmtree(steps_path, ignore_errors=True)
            steps_path.mkdir()
            for obs, action, _, _, done in data.load_data(str(trajectory_path)):
                step_name = f'step{str(step).zfill(5)}.npy'
                step_dict = {'step': step, 'obs': obs, 'action': action, 'done': done}
                np.save(file=steps_path / step_name, arr=step_dict)
                step += 1


if __name__ == "__main__":
    data_dir = '../../code/basalt/data'
    convert_data(data_dir)
