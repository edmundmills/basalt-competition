from agents.soft_q import SoftQAgent
from core.trajectory_generator import TrajectoryGenerator
from core.environment import start_env

import glob
import os
from pathlib import Path
import time

from flatten_dict import unflatten
import gym
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from pyvirtualdisplay import Display
import torch as th
import wandb

# display = Display(visible=0, size=(400, 300))
# display.start()

if __name__ == "__main__":
    run = wandb.init()
    artifact = run.use_artifact(
        'basalt/debug/agent:v68', type='model')
    artifact_dir = artifact.download()
    print(artifact_dir)
    run = artifact.logged_by()
    config = run.config

    cfg = unflatten(config, splitter='dot')
    cfg = OmegaConf.create(cfg)
    print(OmegaConf.to_yaml(cfg))

    # with initialize(config_path='conf'):
    #     cfg = compose('config.yaml')

    cfg.device = "cuda:0" if th.cuda.is_available() else "cpu"
    cfg.wandb = False
    cfg.start_time = time.time()
    cfg.hydra_base_dir = os.getcwd()

    agent_file_path = Path(glob.glob(f"{artifact_dir}/*.pth")[0])
    print(agent_file_path)

    env = start_env(cfg, debug_env=False)
    agent = SoftQAgent(cfg)
    agent.load_parameters(agent_file_path)
    eval_path = Path('eval')
    eval_path.mkdir(exist_ok=True)
    save_path = eval_path / agent_file_path.stem
    save_path.mkdir(exist_ok=True)
    print(save_path)
    generator = TrajectoryGenerator(env, agent, cfg)
    for _ in range(3):
        trajectory = generator.generate(max_episode_length=2000)
        trajectory.save_video(save_path, f'trajectory_{int(round(time.time()))}')
    env.close()
