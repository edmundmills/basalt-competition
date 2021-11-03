import argparse
import os
import time

from flatten_dict import flatten
from hydra import compose, initialize
from omegaconf import OmegaConf
import torch as th


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug-env', dest='debug_env',
                        action='store_true', default=False)
    parser.add_argument('--profile', dest='profile',
                        action='store_true', default=False)
    parser.add_argument('--wandb-false', dest='wandb',
                        action='store_false', default=True)
    parser.add_argument('--virtual-display-false', dest='virtual_display',
                        action='store_false', default=True)
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args()
    return args


def get_config(args):
    with initialize(config_path='../conf'):
        cfg = compose('config.yaml', overrides=args.overrides)

    cfg.device = "cuda:0" if th.cuda.is_available() else "cpu"
    cfg.wandb = args.wandb
    cfg.start_time = time.time()
    if args.profile:
        cfg.env.training_steps = 510
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg

def debug_config(overrides=[]):
    with initialize(config_path='../conf'):
        cfg = compose('config.yaml', overrides=overrides)
    cfg.device = "cuda:0" if th.cuda.is_available() else "cpu"
    cfg.wandb = False
    cfg.start_time = time.time()
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg
