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
obs = {"pov": np.random.randint(0, 255, context.frame_shape),
       "inventory": context.starting_inventory,
       "compassAngle": 0,
       "equipped_items": {"mainhand": {'type': 'snowball'}}}

spatial = th.from_numpy(obs["pov"]).unsqueeze(0)
nonspatial = th.zeros([context.nonspatial_size])
hidden = context.initial_hidden

state_batch = State(spatial.repeat(9, 1, 1, 1),
                    nonspatial.unsqueeze(0).repeat(9, 1),
                    hidden.unsqueeze(0).repeat(9, 1))
action_batch = th.FloatTensor([1]).unsqueeze(0).repeat(9, 1)
reward_batch = th.FloatTensor([0]).unsqueeze(0).repeat(9, 1)
done_batch = th.BoolTensor([False]).unsqueeze(0).repeat(9, 1)
state_sequence = State(spatial.repeat(10, 1, 1, 1).unsqueeze(0),
                       nonspatial.unsqueeze(0).repeat(10, 1).unsqueeze(0),
                       hidden.unsqueeze(0).repeat(10, 1).unsqueeze(0))
action_sequence = th.FloatTensor([1]).unsqueeze(0).repeat(9, 1).unsqueeze(0)
reward_sequence = th.FloatTensor([0]).unsqueeze(0).repeat(9, 1).unsqueeze(0)
done_sequence = th.BoolTensor([False]).unsqueeze(0).repeat(9, 1).unsqueeze(0)


@pytest.fixture
def state():
    return State(spatial, nonspatial, hidden)


@pytest.fixture
def transition():
    return Transition(State(spatial, nonspatial, hidden),
                      th.FloatTensor([1]),
                      th.FloatTensor([0]),
                      State(spatial, nonspatial, hidden),
                      th.BoolTensor([False]))


@pytest.fixture
def sequence():
    return Sequence(state_sequence,
                    action_sequence,
                    reward_sequence,
                    done_sequence)
