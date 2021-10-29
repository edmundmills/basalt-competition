from core.trajectories import Trajectory
from modules.termination_critic import TerminationCritic
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
    for saved_agent_path in reversed(sorted(Path('train/').iterdir())):
        if 'termination_critic' in saved_agent_path.name:
            print(f'Loading {saved_agent_path.name} as termination critic')
            critic.load_parameters(saved_agent_path)
            break
    critic.critique_trajectory(trajectory)
    trajectory.view()
