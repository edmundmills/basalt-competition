from networks.soft_q import SoftQNetwork
from utils.trajectories import Trajectory, TrajectoryGenerator
from utils.environment import start_env
from pyvirtualdisplay import Display

import os
import time
import gym
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch as th

# display = Display(visible=0, size=(400, 300))
# display.start()

if __name__ == "__main__":
    with initialize(config_path='conf'):
        cfg = compose('config.yaml')

    cfg.device = "cuda:0" if th.cuda.is_available() else "cpu"
    cfg.wandb = False
    cfg.start_time = time.time()

    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    environment = cfg.env.name
    training_run = 'MineRLBasaltFindCave-v0_iqlearn_online_1634137333'
    os.environ['MINERL_ENVIRONMENT'] = training_run.split('_')[0]
    env = start_env(debug_env=False)
    model = SoftQNetwork(cfg)
    model_file_name = training_run + '.pth'
    saved_model_path = Path('train') / model_file_name
    model.load_parameters(saved_model_path)
    eval_path = Path('eval')
    eval_path.mkdir(exist_ok=True)
    save_path = eval_path / training_run
    generator = TrajectoryGenerator(env)
    for _ in range(3):
        trajectory = generator.generate(model,
                                        print_actions=True)
        trajectory.save_as_video(save_path, f'trajectory_{int(round(time.time()))}')
    env.close()
