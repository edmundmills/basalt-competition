from networks.soft_q import SoftQNetwork

import os
import time
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch as th


if __name__ == "__main__":
    with initialize(config_path='conf'):
        cfg = compose('config.yaml')

    cfg.device = "cuda:0" if th.cuda.is_available() else "cpu"
    cfg.wandb = False
    cfg.start_time = time.time()

    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    environment = cfg.env.name
    training_run = 'MineRLBasaltMakeWaterfall-v0_iqlearn_online_1634184036'
    os.environ['MINERL_ENVIRONMENT'] = training_run.split('_')[0]
    model = SoftQNetwork(cfg)
    model_file_name = training_run + '.pth'
    saved_model_path = Path('train') / model_file_name
    model.load_parameters(saved_model_path)
    print(model.alpha)
    model.alpha = 0.05832
    print(model.alpha)
    model.save(saved_model_path)
