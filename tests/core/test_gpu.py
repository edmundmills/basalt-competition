from core.gpu import *


class TestNormalizeState:
    def test_valid_state(self, default_config, state_batch):
        gpu_loader = GPULoader(default_config)
        normalized_state_batch = gpu_loader.normalize_state(state_batch)
        assert type(normalized_state_batch) == type(state_batch)


class TestStateToDevice:
    def test_valid_state(self, default_config, state_batch):
        gpu_loader = GPULoader(default_config)
        output_state = gpu_loader.state_to_device(state_batch)
        assert type(output_state) == type(state_batch)


class TestStatesToDevice:
    def test_valid_states(self, default_config, state_batch):
        gpu_loader = GPULoader(default_config)
        all_states = gpu_loader.states_to_device((state_batch, state_batch))
        assert type(all_states[0]) == type(state_batch)


class TestTransitionsToDevice:
    def test_valid_transition(self, default_config, transition_batch):
        gpu_loader = GPULoader(default_config)
        gpu_loader.load_sequences = False
        output_transitions = gpu_loader.transitions_to_device(transition_batch)
        assert type(output_transitions) == type(transition_batch)

    def test_valid_sequence(self, default_config, sequence, transition):
        gpu_loader = GPULoader(default_config)
        output_transitions = gpu_loader.transitions_to_device(sequence)
        assert type(output_transitions) == type(transition)
