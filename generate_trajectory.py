from agents.bc import BCAgent
from helpers.trajectories import Trajectory, TrajectoryGenerator
from helpers.environment import EnvironmentHelper

from pathlib import Path
import os

if __name__ == "__main__":
    MINERL_ENVIRONMENT = 'MineRLBasaltFindCave-v0'
    os.environ['MINERL_ENVIRONMENT'] = MINERL_ENVIRONMENT
    env = EnvironmentHelper.start_env(debug_env=True)
    agent = BCAgent()
    saved_agent_path = Path('train') / 'bc_001.pth'
    agent.load_parameters(saved_agent_path)
    generator = TrajectoryGenerator(env, agent)
    trajectory = generator.generate()
    # trajectory.view()
    save_path = Path('data') / 'eval' / 'run_001'
