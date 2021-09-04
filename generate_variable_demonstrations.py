from agents.bc import BCAgent, NoisyBCAgent
from agents.bcx import BCXAgent, NoisyBCXAgent
from helpers.trajectories import Trajectory, TrajectoryGenerator
from environment.start import start_env
from helpers.data import zip_demonstrations

import argparse
from pathlib import Path
import os
import numpy as np


def generate_variable_demonstrations(agent, demo_count, max_episode_length):
    data_root = Path(os.getenv('MINERL_DATA_ROOT'))
    environment_name = os.getenv('MINERL_ENVIRONMENT')
    env = start_env(debug_env=False)
    generator = TrajectoryGenerator(env, agent)
    noise_schedule = np.linspace(0, 1, demo_count)
    save_paths = []
    for epsilon in noise_schedule:
        print(f'Generating trajectory with epsilon={epsilon}')
        agent.epsilon = epsilon
        trajectory = generator.generate(max_episode_length=max_episode_length)
        print(f'Trajectory completed with {len(trajectory)} steps')
        save_path = data_root / environment_name / f'noisy_bc_{epsilon:.02f}'
        trajectory.save(save_path)
        save_paths.append(save_path)
        print('Trajectory Saved')
        del trajectory
    env.close()
    return save_paths


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--zip-files', dest='zip',
                           action='store_true', default=False)
    args = argparser.parse_args()

    MINERL_ENVIRONMENT = 'MineRLBasaltFindCave-v0'
    os.environ['MINERL_ENVIRONMENT'] = MINERL_ENVIRONMENT
    agent = NoisyBCXAgent(epsilon=0)
    saved_agent_path = Path('train') / '1630688204_bcx.pth'
    agent.load_parameters(saved_agent_path)
    save_paths = generate_variable_demonstrations(agent, 100, 5000)
    if args.zip:
        zip_demonstrations(save_paths, 'demonstrations.zip')
