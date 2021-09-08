from helpers.trajectories import Trajectory
from agents.termination_critic import TerminationCritic
from pathlib import Path
import os

if __name__ == "__main__":
    MINERL_ENVIRONMENT = 'MineRLBasaltFindCave-v0'
    os.environ['MINERL_ENVIRONMENT'] = MINERL_ENVIRONMENT

    MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
    trajectory_path = Path(MINERL_DATA_ROOT) / MINERL_ENVIRONMENT / \
        'v3_organic_gourd_yeti-2_3492-3909'
    trajectory = Trajectory()
    trajectory.load(trajectory_path)
    critic = TerminationCritic()
    saved_agent_path = Path('train') / 'termination_critic_1631119402.pth'
    critic.load_parameters(saved_agent_path)
    critic.critique_trajectory(trajectory)
    trajectory.view()
