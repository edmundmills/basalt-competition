from utils.datasets import ReplayBuffer, SegmentReplayBuffer
from utils.datasets import TrajectoryStepDataset, TrajectorySegmentDataset
from networks.soft_q import SoftQNetwork
from utils.environment import start_env
from utils.trajectories import TrajectoryGenerator
from algorithms.online_imitation import OnlineImitation
from algorithms.iqlearn_sac import IQLearnSAC
from algorithms.curiosity import IntrinsicCuriosityTraining, CuriousIQ

import torch as th
import numpy as np


from pyvirtualdisplay import Display
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from flatten_dict import flatten
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
os.environ["MINERL_DATA_ROOT"] = MINERL_DATA_ROOT


def get_config(args):
    with initialize(config_path='conf'):
        cfg = compose('config.yaml', overrides=args.overrides)

    cfg.device = "cuda:0" if th.cuda.is_available() else "cpu"
    cfg.wandb = args.wandb
    cfg.training_timeout = int(
        os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4 * 24 * 60)) / 4
    if args.profile:
        cfg.method.training_steps = 510
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg


def main():
    """
    This function will be called for training phase.
    This should produce and save same files you upload during your submission.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--debug-env', dest='debug_env',
                           action='store_true', default=False)
    argparser.add_argument('--profile', dest='profile',
                           action='store_true', default=False)
    argparser.add_argument('--wandb-false', dest='wandb',
                           action='store_false', default=True)
    argparser.add_argument('--virtual-display-false', dest='virtual_display',
                           action='store_false', default=True)
    argparser.add_argument("overrides", nargs="*", default=["env=cave"])

    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    config = get_config(args)
    environment = config.env.name
    os.environ['MINERL_ENVIRONMENT'] = environment

    if args.wandb:
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
        env = start_env(debug_env=args.debug_env)
    else:
        env = None

    replay_buffer = ReplayBuffer(config) if config.lstm_layers == 0 \
        else SegmentReplayBuffer(config)
    iter_count = 0
    if config.method.starting_steps > 0:
        replay_buffer = TrajectoryGenerator(
            env, replay_buffer).random_trajectories(
                config.method.starting_steps, lstm_hidden_size=config.lstm_hidden_size)
        iter_count += config.method.starting_steps

    if config.pretraining.name == 'curiosity_pretraining':
        print('Starting Pretraining')
        pretraining_algorithm = IntrinsicCuriosityTraining(
            config, initial_replay_buffer=replay_buffer, initial_iter_count=iter_count)
        pretrained_model, replay_buffer = pretraining_algorithm(env)
        iter_count = pretraining_algorithm.iter_count
        print('Pretraining Completed')
    else:
        print('No Pretraining')
        pretrained_model = None

    # initialize dataset, model, algorithm
    if config.method.expert_dataset:
        if config.lstm_layers == 0:
            expert_dataset = TrajectoryStepDataset(config,
                                                   debug_dataset=args.debug_env)
        else:
            expert_dataset = TrajectorySegmentDataset(config,
                                                      debug_dataset=args.debug_env)

    if pretrained_model is not None:
        model = pretrained_model
    elif config.method.algorithm in ['online_imitation', 'supervised_learning']:
        model = SoftQNetwork(config)

    if config.method.algorithm == 'curious_IQ':
        training_algorithm = CuriousIQ(expert_dataset, config,
                                       initial_replay_buffer=replay_buffer,
                                       initial_iter_count=iter_count)
    elif config.method.algorithm == 'sac' and config.method.loss_function == 'iqlearn':
        training_algorithm = IQLearnSAC(expert_dataset, config,
                                        initial_replay_buffer=replay_buffer,
                                        initial_iter_count=iter_count)
    elif config.method.algorithm == 'online_imitation':
        training_algorithm = OnlineImitation(expert_dataset, model, config,
                                             initial_replay_buffer=replay_buffer,
                                             initial_iter_count=iter_count)
    elif config.method.algorithm == 'supervised_learning':
        training_algorithm = SupervisedLearning(expert_dataset, model, config)

    # run algorithm
    if not args.profile:
        model, replay_buffer = training_algorithm(env)
    else:
        print('Training with profiler')
        profile_dir = f'./logs/{training_algorithm.name}/'
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     on_trace_ready=th.profiler.tensorboard_trace_handler(profile_dir),
                     schedule=schedule(skip_first=32, wait=5,
                     warmup=1, active=3, repeat=2)) as prof:
            with record_function("model_inference"):
                model, replay_buffer = training_algorithm(env, profiler=prof)
            if args.wandb:
                profile_art = wandb.Artifact("trace", type="profile")
                for profile_file_path in Path(profile_dir).iterdir():
                    profile_art.add_file(profile_file_path)
                profile_art.save()

    # save model
    if not args.debug_env:
        model_save_path = os.path.join('train', f'{training_algorithm.name}.pth')
        model.save(model_save_path)
        if args.wandb:
            model_art = wandb.Artifact("agent", type="model")
            model_art.add_file(model_save_path)
            model_art.save()

    # Training 100% Completed
    # aicrowd_helper.register_progress(1)
    if args.virtual_display:
        display.stop()
    if env is not None:
        env.close()


if __name__ == "__main__":
    main()
