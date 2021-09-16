from helpers.training_runs import TrainingRun
from algorithms.sac import SoftActorCritic
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
    argparser.add_argument('--virtual-display', dest='virtual_display',
                           action='store_true', default=False)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    config = dict(
        q_lr=3e-4,
        policy_lr=3e-5,
        curiosity_lr=1e-4,
        starting_steps=300,
        training_steps=5000,
        batch_size=256,
        tau=.05,
        alpha=.01,
        discount_factor=.99,
        n_observation_frames=3,
        environment=environment,
        infra='colab',
        algorithm='curiosity',
    )
    run = TrainingRun(config=config,
                      checkpoint_freqency=1000,
                      wandb=args.wandb)
    config['model_name'] = run.name

    # Start WandB
    if args.wandb:
        wandb.init(
            project="curiosity",
            notes="increase number of frames, batch_size",
            config=config,
        )

    # Start Virual Display
    if args.virtual_display:
        display = Display(visible=0, size=(400, 300))
        display.start()

    # Train Agent
    training_algorithm = SoftActorCritic(run)

    if args.debug_env:
        print('Starting Debug Env')
    else:
        print(f'Starting Env: {environment}')
    env = start_env(debug_env=args.debug_env)

    if not args.profile:
        training_algorithm(env)
    else:
        print('Training with profiler')
        config['training_steps'] = 510
        profile_dir = f'./logs/{run.name}/'
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     on_trace_ready=th.profiler.tensorboard_trace_handler(profile_dir),
                     schedule=schedule(skip_first=32, wait=5,
                     warmup=1, active=3, repeat=2)) as prof:
            with record_function("model_inference"):
                training_algorithm(env, profiler=prof)
            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            if args.wandb:
                profile_art = wandb.Artifact("trace", type="profile")
                for profile_file_path in Path(profile_dir).iterdir():
                    profile_art.add_file(profile_file_path)
                profile_art.save()

    model_save_path = os.path.join('train', f'{run.name}.pth')
    if not args.debug_env:
        training_algorithm.save(model_save_path)
        if args.wandb:
            model_art = wandb.Artifact("agent", type="model")
            model_art.add_file(model_save_path)
            model_art.save()

    # Training 100% Completed
    # aicrowd_helper.register_progress(1)
    if args.virtual_display:
        display.stop()

    env.close()


if __name__ == "__main__":
    main()
