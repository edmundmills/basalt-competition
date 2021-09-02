from environment.debug_env import DebugEnv
from environment.wrappers import ActionShaping

import gym
import minerl
import os


def start_env(debug_env=False):
    if debug_env:
        env = DebugEnv()
    else:
        environment = os.getenv('MINERL_ENVIRONMENT')
        env = gym.make(environment)
        env = ActionShaping(env)
    return env
