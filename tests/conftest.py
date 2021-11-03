from contexts.minerl.environment import MineRLContext
from core.state import State, Transition, Sequence
from utility.config import debug_config

import copy

from argparse import Namespace
import numpy as np
import pytest
import torch as th


@pytest.fixture
def default_args():
    args = Namespace(virtual_display=False, wandb=False, profile=False,
                     debug_env=True, overrides=[])
    return args


@pytest.fixture
def default_config():
    config = debug_config()
    config.method.starting_steps = 100
    config.method.training_steps = 3
    config.method.batch_size = 4
    config.model.lstm_sequence_length = 3
    return config

config = debug_config()
config.method.starting_steps = 100
config.method.training_steps = 3
config.method.batch_size = 4
config.model.lstm_sequence_length = 3
context = MineRLContext(config)
obs = {"pov": np.random.randint(0, 255, context.frame_shape),
       "inventory": context.starting_inventory,
       "compassAngle": 0,
       "equipped_items": {"mainhand": {'type': 'snowball'}}}

spatial = th.from_numpy(obs["pov"]).repeat(
    config.model.n_observation_frames, 1, 1).unsqueeze(0)
nonspatial = th.zeros([context.nonspatial_size])
hidden = context.initial_hidden

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
def state_batch():
    return State(spatial.repeat(9, 1, 1, 1),
                 nonspatial.unsqueeze(0).repeat(9, 1),
                 hidden.unsqueeze(0).repeat(9, 1))


@pytest.fixture
def transition():
    return Transition(State(spatial, nonspatial, hidden),
                      th.FloatTensor([1]),
                      th.FloatTensor([0]),
                      State(spatial, nonspatial, hidden),
                      th.BoolTensor([False]))


@pytest.fixture
def transition_batch():
    return Transition(State(spatial.repeat(9, 1, 1, 1),
                            nonspatial.unsqueeze(0).repeat(9, 1),
                            hidden.unsqueeze(0).repeat(9, 1)),
                      action_batch,
                      reward_batch,
                      State(spatial.repeat(9, 1, 1, 1),
                            nonspatial.unsqueeze(0).repeat(9, 1),
                            hidden.unsqueeze(0).repeat(9, 1)),
                      done_batch)


@pytest.fixture
def sequence():
    return Sequence(state_sequence,
                    action_sequence,
                    reward_sequence,
                    done_sequence)
