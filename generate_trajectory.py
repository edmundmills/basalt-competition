import ast
import glob
from networks.soft_q import SoftQAgent
from core.trajectories import Trajectory, TrajectoryGenerator
from core.environment import start_env
from pyvirtualdisplay import Display
from flatten_dict import unflatten

import os
import time
import gym
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch as th
​import wandb
​
# display = Display(visible=0, size=(400, 300))
# display.start()

if __name__ == "__main__":
    run = wandb.init()
    artifact = run.use_artifact(
        'basalt/MineRLBasaltCreateVillageAnimalPen-v0/agent:v164', type='model')
    artifact_dir = artifact.download()
    print(artifact_dir)
    run = artifact.logged_by()
    config = run.config
​
cfg = unflatten(config, splitter='dot')
cfg = OmegaConf.create(cfg)
print(OmegaConf.to_yaml(cfg))
​
# with initialize(config_path='conf'):
#     cfg = compose('config.yaml')
​
cfg.device = "cuda:0" if th.cuda.is_available() else "cpu"
cfg.wandb = False
cfg.start_time = time.time()
​
cfg.hydra_base_dir = os.getcwd()
environment = cfg.env.name
# FIX: Currently needs to be manually set
cfg.alpha = 3.427
​
# training_run = f'artifacts/artifact.name'
# Get checkpoint from artifact_dir
training_run = glob.glob(f"{artifact_dir}/*.pth")[0][:-4]
print(training_run)
​
os.environ['MINERL_ENVIRONMENT'] = environment
env = start_env(debug_env=False)
model = SoftQAgent(cfg)
model_file_name = training_run + '.pth'
saved_model_path = model_file_name
model.load_parameters(saved_model_path)
eval_path = Path('eval')
eval_path.mkdir(exist_ok=True)
 save_path = eval_path / training_run
  generator = TrajectoryGenerator(env)
   for _ in range(3):
        trajectory = generator.generate(model, max_episode_length=2000,
                                        print_actions=True)
        trajectory.save_as_video(save_path, f'trajectory_{int(round(time.time()))}')
    env.close()
