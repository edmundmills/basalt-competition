from helpers.data import pre_process_expert_trajectories
from generate_trajectories import generate_variable_demonstrations
from helpers.datasets import MultiFrameDataset
from helpers.training_runs import TrainingRun
from agents.bc import BCAgent

import torch as th
import numpy as np

import argparse
import logging
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
    args = argparser.parse_args()

    # Preprocess Data
    if args.preprocess:
        pre_process_expert_trajectories()

    # Train BC
    run = TrainingRun(label='bcx',
                      epochs=2,
                      lr=1e-4)
    dataset = MultiFrameDataset()
    bc_agent = BCXAgent()
    bc_agent.train(dataset, run)

    # Generate variable quality demonstrations
    demo_count = 100
    max_episode_length = 3000
    generate_variable_demonstrations(bc_agent, demo_count, max_episode_length)

    # Train reward model

    # Train offline RL Agent

    # Get feedback

    # Training 100% Completed
    # aicrowd_helper.register_progress(1)
    # env.close()


if __name__ == "__main__":
    main()
