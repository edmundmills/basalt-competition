from networks.soft_q import SoftQNetwork
from helpers.trajectories import Trajectory, TrajectoryGenerator
from environment.start import start_env

from pathlib import Path
import os


if __name__ == "__main__":
    environment = 'MineRLBasaltBuildVillageHouse-v0'
    os.environ['MINERL_ENVIRONMENT'] = environment
    env = start_env(debug_env=False)
    model = SoftQNetwork()
    saved_model_path = Path('train') / \
        'MineRLBasaltBuildVillageHouse-v0_iqlearn_1631558462.pth'
    model.load_parameters(saved_model_path)

    # for saved_agent_path in reversed(sorted(Path('train/').iterdir())):
    #     if ('sqil' in saved_agent_path.name
    #             and environment in saved_agent_path.name):
    #         print(f'Loading {saved_agent_path.name} as agent')
    #         agent.load_parameters(saved_agent_path)
    #         break
    generator = TrajectoryGenerator(env, model)
    trajectory = generator.generate(max_episode_length=2000)
    env.close()
    trajectory.view()
    # save_path = Path('data') / 'eval' / 'run_001'
