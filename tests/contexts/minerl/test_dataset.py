from contexts.minerl.dataset import *
from core.state import State, Transition

import numpy as np


def test_dataset_builder(default_config):
    builder = MineRLDatasetBuilder(default_config, debug_dataset=True)
    trajectories, step_lookup, _ = builder.load_data()
    all_actions = []
    all_rewards = []
    for trajectory in trajectories:
        assert len(trajectory) > 0
        assert len(trajectory.states) == len(trajectory) + 1
        assert type(trajectory.states[0]) == State
        assert type(trajectory.rewards[0]) in [int, np.float32, np.float64]
        assert type(trajectory.done) in [np.bool_, bool]
        all_actions.extend(trajectory.actions)
        all_rewards.extend(trajectory.rewards)
    assert len(step_lookup) == len(all_actions)
    assert len(step_lookup) == len(all_rewards)
    assert all([action in builder.context.actions
               for action in all_actions]) is True
    for trajectory_idx, step_idx in step_lookup:
        transition = trajectories[trajectory_idx][step_idx]
        assert type(transition) == Transition


def test_envs(default_args, default_config):
    envs = ['MineRLBasaltCreateVillageAnimalPen-v0',
            'MineRLBasaltMakeWaterfall-v0',
            'MineRLBasaltBuildVillageHouse-v0',
            'MineRLTreechop-v0']
    for env in envs:
        config = default_config
        config.env.name = env
        builder = MineRLDatasetBuilder(config, debug_dataset=True)
        trajectories, step_lookup, _ = builder.load_data()
        all_actions = []
        all_rewards = []
        for trajectory in trajectories:
            assert len(trajectory) > 0
            assert len(trajectory.states) == len(trajectory) + 1
            assert type(trajectory.states[0]) == State
            assert type(trajectory.rewards[0]) in [int, np.float32, np.float64]
            assert type(trajectory.done) in [np.bool_, bool]
            all_actions.extend(trajectory.actions)
            all_rewards.extend(trajectory.rewards)
        assert len(step_lookup) == len(all_actions)
        assert len(step_lookup) == len(all_rewards)
        assert all([action in builder.context.actions
                   for action in all_actions]) is True
        for trajectory_idx, step_idx in step_lookup:
            transition = trajectories[trajectory_idx][step_idx]
            assert type(transition) == Transition
