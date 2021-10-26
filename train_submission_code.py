from agents.bc import BCAgent
from agents.soft_q import SoftQAgent
import aicrowd_helper
from algorithms.offline import SupervisedLearning
from algorithms.online_imitation import OnlineImitation
from algorithms.sac_demonstrations import SACwithDemonstrations
from algorithms.curiosity import IntrinsicCuriosityTraining, CuriousIQ
from core.datasets import ReplayBuffer, SequenceReplayBuffer
from core.datasets import TrajectoryStepDataset, TrajectorySequenceDataset
from core.environment import start_env
from core.trajectory_generator import TrajectoryGenerator
from networks.termination_critic import TerminationCritic
from utility.config import get_config, parse_args
from utility.parser import Parser

import logging
import os
from pathlib import Path
import time

import coloredlogs
from flatten_dict import flatten
from omegaconf import DictConfig, OmegaConf
from pyvirtualdisplay import Display
import numpy as np
import torch as th
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import wandb

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
parser = Parser(
    'performance/',
    maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
    raise_on_error=False,
    no_entry_poll_timeout=600,
    submission_timeout=MINERL_TRAINING_TIMEOUT * 60,
    initial_poll_timeout=600
)
os.environ["MINERL_DATA_ROOT"] = MINERL_DATA_ROOT


def main(args=None, config=None):
    """
    This function will be called for training phase.
    This should produce and save same files you upload during your submission.
    """
    aicrowd_helper.training_start()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    if not args:
        args = parse_args()
    if not config:
        config = get_config(args)

    environment = config.env.name

    if config.wandb:
        wandb.init(
            project=config.project_name,
            entity="basalt",
            notes="test",
            config=flatten(OmegaConf.to_container(config, resolve=True),
                           reducer='dot'),
        )

    # Start Virual Display
    if args.virtual_display:
        display = Display(visible=0, size=(400, 300))
        display.start()

    # Start env
    if config.method.algorithm != 'supervised_learning':
        if args.debug_env:
            print('Starting Debug Env')
        else:
            print(f'Starting Env: {environment}')
        env = start_env(config, debug_env=args.debug_env)
    else:
        env = None

    replay_buffer = ReplayBuffer(config) if config.lstm_layers == 0 \
        else SequenceReplayBuffer(config)
    iter_count = 0
    if config.method.starting_steps > 0:
        replay_buffer = TrajectoryGenerator(
            env, None, config, replay_buffer,
        ).random_trajectories(config.method.starting_steps)
        iter_count += config.method.starting_steps

    # initialize dataset, agent, algorithm
    if config.method.expert_dataset:
        if config.lstm_layers == 0:
            expert_dataset = TrajectoryStepDataset(config,
                                                   debug_dataset=args.debug_env)
        else:
            expert_dataset = TrajectorySequenceDataset(config,
                                                       debug_dataset=args.debug_env)

    if config.method.algorithm in ['online_imitation']:
        agent = SoftQAgent(config)
    elif config.method.algorithm == 'supervised_learning':
        agent = BCAgent(config)

    # if config.method.algorithm == 'curious_IQ':
    #     training_algorithm = CuriousIQ(expert_dataset, config,
    #                                    initial_replay_buffer=replay_buffer,
    #                                    initial_iter_count=iter_count)
    if config.method.algorithm == 'sac' and config.method.loss_function == 'iqlearn':
        training_algorithm = SACwithDemonstrations(expert_dataset, config,
                                                   initial_replay_buffer=replay_buffer,
                                                   initial_iter_count=iter_count)
    if config.method.algorithm == 'online_imitation':
        training_algorithm = OnlineImitation(expert_dataset, agent, config,
                                             initial_replay_buffer=replay_buffer,
                                             initial_iter_count=iter_count)
    elif config.method.algorithm == 'supervised_learning':
        training_algorithm = SupervisedLearning(expert_dataset, agent, config)

    # run algorithm
    if not args.profile:
        agent, replay_buffer = training_algorithm(env)
    else:
        print('Training with profiler')
        profile_dir = f'./logs/{training_algorithm.name}/'
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     on_trace_ready=th.profiler.tensorboard_trace_handler(profile_dir),
                     schedule=schedule(skip_first=32, wait=5,
                     warmup=1, active=3, repeat=2)) as prof:
            with record_function("model_inference"):
                agent, replay_buffer = training_algorithm(env, profiler=prof)
            if args.wandb:
                profile_art = wandb.Artifact("trace", type="profile")
                for profile_file_path in Path(profile_dir).iterdir():
                    profile_art.add_file(profile_file_path)
                profile_art.save()

    # save model
    if not args.debug_env:
        agent_save_path = os.path.join('train', f'{training_algorithm.name}.pth')
        agent.save(agent_save_path)
        if args.wandb:
            model_art = wandb.Artifact("agent", type="model")
            model_art.add_file(agent_save_path)
            model_art.save()

    # Training 100% Completed
    aicrowd_helper.register_progress(1)
    if args.virtual_display:
        display.stop()
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
