from core.datasets import *
from core.state import Transition, Sequence
from utility.config import parse_args, get_config


class TestTrajectoryStepDataset:
    def test_dataset_with_no_lstm(self):
        args = parse_args()
        args.virtual_display = False
        args.debug_env = True
        args.wandb = False
        args.overrides.append('model=base')
        config = get_config(args)
        dataset = TrajectoryStepDataset(config, debug_dataset=True)
        assert len(dataset) > 0
        assert len(dataset.trajectories) > 0
        assert sum([len(trajectory) for trajectory in dataset.trajectories]) \
            == len(dataset.step_lookup)
        for idx, (sample, master_idx) in enumerate(dataset):
            assert master_idx == idx
            assert type(sample) == Transition


class TestTrajectorySequenceDataset:
    def test_dataset_with_lstm(self):
        args = parse_args()
        args.virtual_display = False
        args.debug_env = True
        args.wandb = False
        args.overrides.append('model=lstm')
        config = get_config(args)
        lstm_sequence_length = 5
        config.model.lstm_sequence_length = lstm_sequence_length
        dataset = TrajectorySequenceDataset(config, debug_dataset=True)
        assert len(dataset) > 0
        assert len(dataset.trajectories) > 0
        sequence_counts = [max(0, len(trajectory) - (lstm_sequence_length - 1))
                           for trajectory in dataset.trajectories]
        assert sum(sequence_counts) == len(dataset.sequence_lookup)
        assert sum(sequence_counts) == len(dataset.active_lookup)
        for idx, (sample, master_idx) in enumerate(dataset):
            assert master_idx == idx
            assert type(sample) == Sequence
            assert sample.states.spatial.size()[0] == lstm_sequence_length + 1
            assert sample.rewards.size()[0] == lstm_sequence_length
            assert sample.actions.size()[0] == lstm_sequence_length
