from agents.bc import BCAgent, NoisyBCAgent
from agents.bcx import BCXAgent, NoisyBCXAgent
from helpers.trajectories import Trajectory, TrajectoryGenerator
from environment.start import start_env

from pathlib import Path
import os

if __name__ == "__main__":
    MINERL_ENVIRONMENT = 'MineRLBasaltFindCave-v0'
    os.environ['MINERL_ENVIRONMENT'] = MINERL_ENVIRONMENT
    env = start_env(debug_env=False)
    agent = NoisyBCXAgent(epsilon=0.2)
    saved_agent_path = Path('train') / 'bcx_001.pth'
    agent.load_parameters(saved_agent_path)
    generator = TrajectoryGenerator(env, agent)
    trajectory = generator.generate()
    trajectory.view()
    save_path = Path('data') / 'eval' / 'run_001'
    del trajectory
    env.close()
