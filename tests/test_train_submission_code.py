from train_submission_code import *
from utility.config import debug_config

import copy


class TestIntegration:
    def test_default_config(self, default_args, default_config):
        main(default_args, default_config)

    class TestMethods:
        def test_sac(self, default_args):
            args = default_args
            config = debug_config(['method=iqlearn_sac'])
            config.method.starting_steps = 100
            config.method.training_steps = 3
            config.method.batch_size = 4
            config.model.lstm_sequence_length = 3
            main(args, config)

        def test_bc(self, default_args):
            args = default_args
            config = debug_config(['method=bc'])
            config.method.max_training_steps = 3
            config.method.batch_size = 4
            config.model.lstm_sequence_length = 3
            main(args, config)

        def test_iqlearn_offline(self, default_args):
            args = default_args
            config = debug_config(['method=iqlearn_offline'])
            config.method.max_training_steps = 3
            config.method.batch_size = 4
            config.model.lstm_sequence_length = 3
            main(args, config)

    class TestModels:
        def test_no_lstm(self, default_args):
            args = default_args
            config = debug_config(['model=base'])
            config.method.starting_steps = 100
            config.method.training_steps = 3
            config.method.batch_size = 4
            main(args, config)

    class TestEnvs:
        def test_waterfall(self, default_args, default_config):
            config = default_config
            config.env.name = 'MineRLBasaltMakeWaterfall-v0'
            main(default_args, config)

        def test_pen(self, default_args, default_config):
            config = default_config
            config.env.name = 'MineRLBasaltCreateVillageAnimalPen-v0'
            main(default_args, config)

        def test_house(self, default_args, default_config):
            config = default_config
            config.env.name = 'MineRLBasaltBuildVillageHouse-v0'
            main(default_args, config)

        def test_treechop(self, default_args, default_config):
            config = default_config
            config.env.name = 'MineRLTreechop-v0'
            main(default_args, config)

        # def test_navigate(self, default_args, default_config):
        #     config = default_config
        #     config.env.name = 'MineRLNavigateDense-v0'
        #     main(default_args, config)
