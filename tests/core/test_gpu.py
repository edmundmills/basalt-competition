from core.gpu import *


class TestNormalizeState:
    def test_valid_state(self, config, state_batch):
        gpu_loader = GPULoader(config)
        normalized_state_batch = gpu_loader.normalize_state(state_batch)
        assert type(normalized_state_batch) == type(state_batch)
