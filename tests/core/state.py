from core.state import *


def test_cat_states(state):
    state1 = state
    state2 = state
    all_states, state_lengths = cat_states((state1, state2))
    assert len(all_states) == 3
    assert len(state_lengths) == 2
    assert state_lengths[0] == state1.spatial.size()[0]
    assert state_lengths[1] == state2.spatial.size()[0]
