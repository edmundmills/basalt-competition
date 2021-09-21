from networks.soft_q import SoftQNetwork
from helpers.trajectories import Trajectory, TrajectoryGenerator
from environment.start import start_env

from pathlib import Path
import os
import time

if __name__ == "__main__":
    environment = 'MineRLBasaltFindCave-v0'
    os.environ['MINERL_ENVIRONMENT'] = environment
    env = start_env(debug_env=False)
    model = SoftQNetwork(alpha=.1, n_observation_frames=3)
    training_run = 'MineRLBasaltFindCave-v0_online_imitation_1632236774'
    saved_model_path = Path('train') / training_run / 'model.pth'
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
    eval_path = Path('eval')
    eval_path.mkdir(exist_ok=True)
    save_path = eval_path / training_run
    trajectory.save_as_video(save_path, f'trajectory_{int(round(time.time()))}')
    trajectory.view()
