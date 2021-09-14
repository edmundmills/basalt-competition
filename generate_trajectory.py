from agents.bc import BCAgent
from agents.soft_q import SqilAgent, IQLearnAgent, SoftQAgent
from helpers.trajectories import Trajectory, TrajectoryGenerator
from environment.start import start_env

from pathlib import Path
import os


if __name__ == "__main__":
    environment = 'MineRLBasaltBuildVillageHouse-v0'
    os.environ['MINERL_ENVIRONMENT'] = environment
    env = start_env(debug_env=False)
    agent = SoftQAgent()
    saved_agent_path = Path('train') / \
        'MineRLBasaltBuildVillageHouse-v0_iqlearn_1631558462.pth'
    agent.load_parameters(saved_agent_path)

    # for saved_agent_path in reversed(sorted(Path('train/').iterdir())):
    #     if ('sqil' in saved_agent_path.name
    #             and environment in saved_agent_path.name):
    #         print(f'Loading {saved_agent_path.name} as agent')
    #         agent.load_parameters(saved_agent_path)
    #         break
    generator = TrajectoryGenerator(env, agent)
    trajectory = generator.generate(max_episode_length=2000)
    env.close()
    trajectory.view()
    # save_path = Path('data') / 'eval' / 'run_001'
