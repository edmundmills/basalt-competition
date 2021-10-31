from core.trajectories import *

import pytest


class TestIndexing:
    def test_initial_trajectory(self, state):
        trajectory = Trajectory()
        trajectory.states.append(state)
        assert len(trajectory) == 0
        with pytest.raises(IndexError) as e_info:
            trajectory[0]

    def test_not_done_trajectory(self, state, transition):
        trajectory = Trajectory()
        trajectory.states.append(state)
        action = transition.action.item()
        reward = transition.reward.item()
        next_state = transition.next_state
        done = transition.done.item()
        trajectory.append_step(action, reward, next_state, done)
        trajectory.append_step(action, reward, next_state, done)
        assert len(trajectory) == 2
        assert trajectory.done is False
        assert trajectory[0].state == state
        assert trajectory.states[2] == next_state
        with pytest.raises(IndexError) as e_info:
            trajectory[2]

    def test_done_trajectory(self, state, transition):
        trajectory = Trajectory()
        trajectory.states.append(state)
        action = transition.action.item()
        reward = transition.reward.item()
        next_state = transition.next_state
        done = transition.done.item()
        trajectory.append_step(action, reward, next_state, done)
        trajectory.append_step(action, reward, next_state, True)
        assert len(trajectory) == 2
        assert trajectory.done is True
        assert trajectory.states[2] == next_state
        assert trajectory[0].state == state
        with pytest.raises(IndexError) as e_info:
            trajectory[2]


class TestGetSequence:
    def test_valid_sequence(self, state, transition):
        trajectory = Trajectory()
        trajectory.states.append(state)
        action = transition.action.item()
        reward = transition.reward.item()
        next_state = transition.next_state
        done = transition.done.item()
        trajectory.append_step(action, reward, next_state, done)
        trajectory.append_step(action, reward, next_state, done)
        trajectory.append_step(action, reward, next_state, done)
        trajectory.append_step(action, reward, next_state, done)
        assert len(trajectory) == 4
        assert trajectory.done is False
        sequence_length = 4
        idx = 3
        sequence = trajectory.get_sequence(idx, sequence_length)
        assert type(sequence) == Sequence
        assert sequence.dones.size()[0] == sequence_length
        assert sequence.rewards.size()[0] == sequence_length
        assert sequence.actions.size()[0] == sequence_length
        assert sequence.states.spatial.size()[0] == sequence_length + 1
        assert sequence.dones[-1] == 0

    def test_too_long_sequence(self, state, transition):
        trajectory = Trajectory()
        trajectory.states.append(state)
        action = transition.action.item()
        reward = transition.reward.item()
        next_state = transition.next_state
        done = transition.done.item()
        trajectory.append_step(action, reward, next_state, done)
        trajectory.append_step(action, reward, next_state, done)
        trajectory.append_step(action, reward, next_state, done)
        trajectory.append_step(action, reward, next_state, done)
        assert len(trajectory) == 4
        sequence_length = 5
        idx = 3
        with pytest.raises(IndexError) as e_info:
            trajectory.get_sequence(idx, sequence_length)

    def test_valid_long_sequence_too_early(self, state, transition):
        trajectory = Trajectory()
        trajectory.states.append(state)
        action = transition.action.item()
        reward = transition.reward.item()
        next_state = transition.next_state
        done = transition.done.item()
        trajectory.append_step(action, reward, next_state, done)
        trajectory.append_step(action, reward, next_state, done)
        trajectory.append_step(action, reward, next_state, done)
        trajectory.append_step(action, reward, next_state, done)
        assert len(trajectory) == 4
        sequence_length = 4
        idx = 2
        with pytest.raises(IndexError) as e_info:
            trajectory.get_sequence(idx, sequence_length)

    def test_idx_out_of_range(self, state, transition):
        trajectory = Trajectory()
        trajectory.states.append(state)
        action = transition.action.item()
        reward = transition.reward.item()
        next_state = transition.next_state
        done = transition.done.item()
        trajectory.append_step(action, reward, next_state, done)
        trajectory.append_step(action, reward, next_state, done)
        trajectory.append_step(action, reward, next_state, done)
        trajectory.append_step(action, reward, next_state, done)
        assert len(trajectory) == 4
        sequence_length = 3
        idx = 4
        with pytest.raises(IndexError) as e_info:
            trajectory.get_sequence(idx, sequence_length)


class TestAdditionalData:
    def test_always_with_key(self, state, transition):
        trajectory = Trajectory()
        trajectory.states.append(state)
        action = transition.action.item()
        reward = transition.reward.item()
        next_state = transition.next_state
        done = transition.done.item()
        trajectory.append_step(action, reward, next_state, done,
                               voluntary_termination=False)
        trajectory.append_step(action, reward, next_state, done,
                               voluntary_termination=False)
        trajectory.append_step(action, reward, next_state, done,
                               voluntary_termination=False)
        trajectory.append_step(action, reward, next_state, done,
                               voluntary_termination=False)
        assert trajectory.additional_step_data['voluntary_termination'] \
            == [False, False, False, False]

    def test_always_with_key(self, state, transition):
        trajectory = Trajectory()
        trajectory.states.append(state)
        action = transition.action.item()
        reward = transition.reward.item()
        next_state = transition.next_state
        done = transition.done.item()
        trajectory.append_step(action, reward, next_state, done,
                               voluntary_termination=False)
        trajectory.append_step(action, reward, next_state, done,
                               voluntary_termination=False)
        trajectory.append_step(action, reward, next_state, done,
                               voluntary_termination=False)
        trajectory.append_step(action, reward, next_state, done,
                               voluntary_termination=False)
        assert trajectory.additional_step_data[3] == {'voluntary_termination': False}
