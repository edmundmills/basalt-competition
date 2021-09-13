from agents.bc import BCAgent
from agents.soft_q import SqilAgent
from helpers.trajectories import Trajectory, TrajectoryGenerator
from environment.start import start_env

from pathlib import Path
import os


if __name__ == "__main__":
    MINERL_ENVIRONMENT = 'MineRLBasaltBuildVillageHouse-v0'
    os.environ['MINERL_ENVIRONMENT'] = MINERL_ENVIRONMENT
    env = start_env(debug_env=False)
    agent = SqilAgent()
    # saved_agent_path = Path('train') / \
    #     'MineRLBasaltBuildVillageHouse-v0_sqil_1631549741.pth'
    for saved_agent_path in reversed(sorted(Path('train/').iterdir())):
        if ('sqil' in saved_agent_path.name
                and environment in saved_agent_path.name):
            print(f'Loading {saved_agent_path.name} as agent')
            agent.load_parameters(saved_agent_path)
            break
    generator = TrajectoryGenerator(env, agent)
    trajectory = generator.generate(max_episode_length=2000)
    env.close()
    trajectory.view()
    # save_path = Path('data') / 'eval' / 'run_001'
