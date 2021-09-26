from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from helpers.datasets import TrajectoryStepDataset
from algorithms.supervised import SupervisedLearning
from networks.bc import BC
from environment.start import start_env

import torch as th
import numpy as np


from pyvirtualdisplay import Display
import wandb
from pathlib import Path
import argparse
import logging
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import os
# import aicrowd_helper
# from utility.parser import Parser
import coloredlogs
coloredlogs.install(logging.DEBUG)


def get_config(args):
    with initialize(config_path='conf'):
        cfg = compose('config.yaml', overrides=args.overrides)

    cfg.device = "cuda:0" if th.cuda.is_available() else "cpu"
    cfg.wandb = args.wandb
    cfg.method.wandb = args.wandb
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--debug-env', dest='debug_env',
                           action='store_true', default=False)
    argparser.add_argument('--wandb-false', dest='wandb',
                           action='store_false', default=True)
    argparser.add_argument("overrides", nargs="*", default=["env=cave", "method=bc"])

    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    config = get_config(args)
    environment = config.env.name
    os.environ['MINERL_ENVIRONMENT'] = environment

    # Start WandB
    if args.wandb:
        wandb.init(
            project="observation space",
            notes="test",
            config=config,
        )

    # Train Agent
    training_algorithm = SupervisedLearning(config.method)
    model = BC(n_observation_frames=config.n_observation_frames)
    # set up dataset
    dataset = TrajectoryStepDataset(
        debug_dataset=args.debug_env, n_observation_frames=config.n_observation_frames)
    train_dataset_size = int(0.8 * len(dataset))
    train_dataset, test_dataset \
        = th.utils.data.random_split(dataset,
                                     [train_dataset_size, len(
                                         dataset) - train_dataset_size],
                                     generator=th.Generator().manual_seed(42))
    print('Train dataset size: ', len(train_dataset))
    print('Test dataset size: ', len(test_dataset))
    print('Training agent...')
    training_algorithm(model,
                       train_dataset,
                       test_dataset=test_dataset)


if __name__ == "__main__":
    main()
