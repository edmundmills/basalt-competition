from core.state import *


class TestCatStates:
    def test_valid_states(self, state):
        state1 = state
        state2 = state
        all_states, state_lengths = cat_states((state1, state2))
        assert len(all_states) == 3
        assert len(state_lengths) == 2
        assert state_lengths[0] == state1.spatial.size()[0]
        assert state_lengths[1] == state2.spatial.size()[0]
        for component1, component2 in zip(state1, all_states):
            assert type(component1) == type(component2)


class TestCatTransitions:
    def test_valid_transitions(self, transition):
        transition1 = transition
        transition2 = transition
        transitions = cat_transitions((transition1, transition2))
        assert len(transitions) == 5
        assert transitions.action.size()[0] == \
            transition1.action.size()[0] + transition2.action.size()[0]
        for component1, component2 in zip(transitions, transition1):
            assert type(component1) == type(component2)


class TestSequenceToTransitions:
    def test_valid_transition(self, sequence, transition):
        transitions = sequence_to_transitions(sequence)
        assert transitions.state.spatial.size()[1] == \
            sequence.states.spatial.size()[1] - 1
        assert transitions.action.size()[0] == sequence.actions.size()[0]
        assert transitions.reward.size()[0] == sequence.rewards.size()[0]
        assert transitions.done.size()[0] == sequence.dones.size()[0]
        for component1, component2 in zip(transition, transitions):
            assert type(component1) == type(component2)
