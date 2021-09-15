from helpers.datasets import TrajectoryStepDataset
from helpers.training_runs import TrainingRun
from networks.soft_q import SqilAgent, IQLearnAgent
from networks.termination_critic import TerminationCritic
from environment.start import start_env
from algorithms.online_imitation import OnlineImitation

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
    environment = 'MineRLBasaltFindCave-v0'
    os.environ['MINERL_ENVIRONMENT'] = environment

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--termination-critic', dest='termination_critic',
                           action='store_true', default=False)
    argparser.add_argument('--train-critic-false', dest='train_critic',
                           action='store_false', default=True)
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
        learning_rate=3e-5,
        training_steps=1000,
        batch_size=64,
        alpha=.01,
        discount_factor=0.99,
        n_observation_frames=3,
        environment=environment,
        infra='colab',
        loss_function='iqlearn',
    )
    run = TrainingRun(config=config,
                      checkpoint_freqency=1000,
                      wandb=args.wandb)
    config['model_name'] = run.name

    if args.wandb:
        wandb.init(
            project="optimize training",
            notes="increase number of frames, batch_size",
            config=config,
        )

    expert_dataset = TrajectoryStepDataset(
        debug_dataset=args.debug_env, n_observation_frames=config['n_observation_frames'])

    # Train termination critic
    if args.termination_critic:
        critic = TerminationCritic()
        if args.train_critic:
            critic_config = dict(algorithm='termination_critic',
                                 epochs=5,
                                 learning_rate=1e-4,
                                 batch_size=32,
                                 environment=environment)
            critic_run = TrainingRun(config=critic_config)

            expert_dataset.n_observation_frames = 1
            critic.train(expert_dataset, critic_run)
            expert_dataset.n_observation_frames = config['n_observation_frames']
        else:
            for saved_agent_path in reversed(sorted(Path('train/').iterdir())):
                if ('termination_critic' in saved_agent_path.name
                        and environment in saved_agent_path.name):
                    print(f'Loading {saved_agent_path.name} as termination critic')
                    critic.load_parameters(saved_agent_path)
                    break
    else:
        critic = None

    # Start Virual Display
    if args.virtual_display:
        display = Display(visible=0, size=(400, 300))
        display.start()

    # Train Agent
    training_algorithm = OnlineImitation(loss_function=config['loss_function'],
                                         termination_critic=critic)
    model = SoftQNetwork(alpha=config['alpha'],
                         n_observation_frames=config['n_observation_frames'])
    if args.debug_env:
        print('Starting Debug Env')
    else:
        print(f'Starting Env: {environment}')
    env = start_env(debug_env=args.debug_env)
    if args.profile:
        print('Training with profiler')
        config['training_steps'] = 510
        profile_dir = f'./logs/{run.name}/'
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     on_trace_ready=th.profiler.tensorboard_trace_handler(profile_dir),
                     schedule=schedule(skip_first=32, wait=5,
                     warmup=1, active=3, repeat=2)) as prof:
            with record_function("model_inference"):
                training_algorithm(model, env, expert_dataset, run, profiler=prof)
            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            if args.wandb:
                profile_art = wandb.Artifact("trace", type="profile")
                for profile_file_path in Path(profile_dir).iterdir():
                    profile_art.add_file(profile_file_path)
                profile_art.save()

    else:
        training_algorithm(model, env, expert_dataset, run)
    if not args.debug_env:
        model_save_path = os.path.join('train', f'{run.name}.pth')
        agent.save(model_save_path)
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
