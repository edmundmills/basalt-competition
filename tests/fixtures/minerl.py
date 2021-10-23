from core.state import State, Transition, Sequence
from contexts.minerl.environment import MineRLContext
from utility.get_config import get_config, parse_args

from collections import namedtuple

import numpy as np
import pytest
import torch as th

args = parse_args()
config = get_config(args)

context = MineRLContext(config)
obs = {"pov": np.random.randint(0, 255, (64, 64, 3)),
       "inventory": context.starting_inventory,
       "compassAngle": 0,
       "equipped_items": {"mainhand": {'type': 'snowball'}}}

spatial = th.from_numpy(obs["pov"])
nonspatial = th.zeros([2])


@pytest.fixture
def state():
    return State(spatial,
                 nonspatial,
                 context.initial_hidden)


@pytest.fixture
def action():
    return -1


@pytest.fixture
def reward():
    return 0


@pytest.fixture
def done():
    return False


@pytest.fixture
def transition():
    return Transition(state, action, reward, state, done)
