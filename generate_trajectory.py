from agents.bc import BCAgent
from helpers.trajectories import Trajectory, TrajectoryGenerator
from helpers.environment import EnvironmentHelper

from pathlib import Path

if __name__ == "__main__":
    env = EnvironmentHelper.start_env(debug_env=True)
    agent = BCAgent()
    saved_agent_path = Path('train') / 'bc_001.pth'
    agent.load_parameters()
    generator = TrajectoryGenerator(env, agent)
    save_path = Path('data') / 'eval' / 'run_001'
    trajectory = generator.generate(save_path)
    # trajectory.view()
