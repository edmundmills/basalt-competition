from helpers.data import pre_process_expert_trajectories
from helpers.datasets import MultiFrameDataset
from agents.bc import BCAgent

import torch as th
import numpy as np

import logging
import os
# import aicrowd_helper
# from utility.parser import Parser
import coloredlogs
coloredlogs.install(logging.DEBUG)


# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
# You need to ensure that your submission is trained within allowed training time.
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4 * 24 * 60))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))

# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on.
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
    RUN_NAME = '001'
    os.environ['MINERL_ENVIRONMENT'] = MINERL_ENVIRONMENT
    os.environ['TRAIN_PATH'] = 'train/'
    os.environ['RUN_NAME'] = RUN_NAME

    # Preprocess Data
    # pre_process_expert_trajectories()

    # Train BC
    epochs = 1
    lr = 1e-4
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    dataset = MultiFrameDataset()
    bc_agent = BCAgent(device=device)
    bc_agent.train(dataset, epochs, lr)

    # Generate variable quality demonstrations

    # Train reward model

    # Train offline RL Agent

    # Get feedback

    # Training 100% Completed
    # aicrowd_helper.register_progress(1)
    env.close()


if __name__ == "__main__":
    main()
