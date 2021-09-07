from helpers.trajectories import Trajectory
from agents.termination_critic import TerminationCritic
from pathlib import Path
import os

if __name__ == "__main__":
    MINERL_ENVIRONMENT = 'MineRLBasaltFindCave-v0'
    os.environ['MINERL_ENVIRONMENT'] = MINERL_ENVIRONMENT

    MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
    trajectory_path = Path(MINERL_DATA_ROOT) / MINERL_ENVIRONMENT / \
        'v3_accomplished_pattypan_squash_ghost-6_1739-2809'
    trajectory = Trajectory()
    trajectory.load(trajectory_path)
    critic = TerminationCritic()
    saved_agent_path = Path('train') / 'termination_critic_1630986816.pth'
    critic.load_parameters(saved_agent_path)
    critic.critique_trajectory(trajectory)
    trajectory.view()
