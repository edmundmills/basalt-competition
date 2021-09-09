from helpers.data import pre_process_expert_trajectories
from helpers.datasets import StepDataset, MultiFrameDataset
from helpers.training_runs import TrainingRun
from agents.bc import BCAgent
from agents.soft_q import SqilAgent
from agents.termination_critic import TerminationCritic
from environment.start import start_env

import torch as th
import numpy as np

from pathlib import Path
import argparse
import logging
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import os
# import aicrowd_helper
# from utility.parser import Parser
import coloredlogs
coloredlogs.install(logging.DEBUG)


# You need to ensure that your submission is trained by launching less
# than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
# You need to ensure that your submission is trained within allowed training time.
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4 * 24 * 60))
# You need to ensure that your submission is trained by launching
# less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))

# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched
# and so on.
# Make your you keep a tap on the numbers to avoid breaching any limits.
# parser = Parser(
#     'performance/',
#     maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
#     raise_on_error=False,
#     no_entry_poll_timeout=600,
#     submission_timeout=MINERL_TRAINING_TIMEOUT * 60,
#     initial_poll_timeout=600
# )


def main():
    """
    This function will be called for training phase.
    This should produce and save same files you upload during your submission.
    """
    MINERL_ENVIRONMENT = 'MineRLBasaltFindCave-v0'
    os.environ['MINERL_ENVIRONMENT'] = MINERL_ENVIRONMENT

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--preprocess-false', dest='preprocess',
                           action='store_false', default=True)
    argparser.add_argument('--train-critic-false', dest='train_critic',
                           action='store_false', default=True)
    argparser.add_argument('--debug-env', dest='debug_env',
                           action='store_true', default=False)
    argparser.add_argument('--profile', dest='profile',
                           action='store_true', default=False)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    # Preprocess Data
    if args.preprocess:
        pre_process_expert_trajectories()

    # Train termination critic
    critic = TerminationCritic()
    if args.train_critic:
        run = TrainingRun(label='termination_critic',
                          epochs=5,
                          lr=1e-4)
        dataset = StepDataset()
        critic.train(dataset, run)
    else:
        for saved_agent_path in reversed(sorted(Path('train/').iterdir())):
            if 'termination_critic' in saved_agent_path.name:
                print(f'Loading {saved_agent_path.name} as termination critic')
                critic.load_parameters(saved_agent_path)
                break

    # Train Agent
    run = TrainingRun(label='sqil',
                      training_steps=20000,
                      lr=1e-4,
                      discount_factor=0.99)
    bc_agent = SqilAgent(termination_critic=critic)
    if args.debug_env:
        print('Starting Debug Env')
    env = start_env(debug_env=args.debug_env)
    if args.profile:
        print('Training with profiler')
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     schedule=schedule(skip_first=10, wait=5,
                     warmup=1, active=3, repeat=2)) as prof:
            with record_function("model_inference"):
                bc_agent.train(env, run, profiler=prof)
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    else:
        bc_agent.train(env, run)

    # Training 100% Completed
    # aicrowd_helper.register_progress(1)


if __name__ == "__main__":
    main()
