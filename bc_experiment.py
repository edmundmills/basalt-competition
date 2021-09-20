from algorithms.supervised import SupervisedLearning
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


def main():
    environment = 'MineRLBasaltFindCave-v0'
    os.environ['MINERL_ENVIRONMENT'] = environment

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--debug-env', dest='debug_env',
                           action='store_true', default=False)
    argparser.add_argument('--profile', dest='profile',
                           action='store_true', default=False)
    argparser.add_argument('--wandb', dest='wandb',
                           action='store_true', default=False)

    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    config = dict(
        learning_rate=.001,
        epochs=5,
        batch_size=64,
        n_observation_frames=3,
        environment=environment,
        algorithm='supervised_learning',
        wandb=args.wandb
    )

    # Start WandB
    if args.wandb:
        wandb.init(
            project="observation space",
            notes="test",
            config=config,
        )

    # Train Agent
    training_algorithm = SupervisedLearning(config)

    # set up dataset
    dataset = TrajectoryStepDataset(
        debug_dataset=args.debug_env, n_observation_frames=config['n_observation_frames'])
    train_dataset_size = int(0.8 * len(dataset))
    train_dataset, test_dataset \
        = th.utils.data.random_split(dataset,
                                     [train_dataset_size, len(
                                         dataset) - train_dataset_size],
                                     generator=th.Generator().manual_seed(42))

    if not args.profile:
        training_algorithm(model,
                           train_dataset,
                           test_dataset=test_dataset)
    else:
        print('Training with profiler')
        config['training_steps'] = 510
        profile_dir = f'./logs/{training_algorithm.name}/'
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     on_trace_ready=th.profiler.tensorboard_trace_handler(profile_dir),
                     schedule=schedule(skip_first=32, wait=5,
                     warmup=1, active=3, repeat=2)) as prof:
            with record_function("model_inference"):
                training_algorithm(model,
                                   train_dataset,
                                   test_dataset=test_dataset,
                                   profiler=prof)
            if args.wandb:
                profile_art = wandb.Artifact("trace", type="profile")
                for profile_file_path in Path(profile_dir).iterdir():
                    profile_art.add_file(profile_file_path)
                profile_art.save()

    if not args.debug_env:
        model_save_path = os.path.join('train', f'{training_algorithm.name}.pth')
        training_algorithm.save(model_save_path)

    # Training 100% Completed
    # aicrowd_helper.register_progress(1)


if __name__ == "__main__":
    main()
