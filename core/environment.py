import gym
import minerl

import os
import copy
from collections import OrderedDict
import random

import numpy as np
import torch as th
import torch.nn.functional as F


def start_env(debug_env=False):
    if debug_env:
        env = DebugEnv()
    else:
        environment = os.getenv('MINERL_ENVIRONMENT')
        env = gym.make(environment)
        env = ActionShaping(env)
    return env
