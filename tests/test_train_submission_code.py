from train_submission_code import *
from utility.config import get_config, parse_args

import copy


class TestIntegration:
    def test_default_config(self, default_args, default_config):
        main(default_args, default_config)

    class TestMethods:
        def test_sac(self):
            args = parse_args()
            args.virtual_display = False
            args.debug_env = True
            args.wandb = False
            args.overrides.append('method=iqlearn_sac')
            config = get_config(args)
            config.method.starting_steps = 100
            config.method.training_steps = 3
            config.method.batch_size = 4
            config.model.lstm_sequence_length = 3
            main(args, config)

        def test_bc(self):
            args = parse_args()
            args.virtual_display = False
            args.debug_env = True
            args.wandb = False
            args.overrides.append('method=bc')
            config = get_config(args)
            config.method.max_training_steps = 3
            config.method.batch_size = 4
            config.model.lstm_sequence_length = 3
            main(args, config)

        def test_iqlearn_offline(self):
            args = parse_args()
            args.virtual_display = False
            args.debug_env = True
            args.wandb = False
            args.overrides.append('method=iqlearn_offline')
            config = get_config(args)
            config.method.max_training_steps = 3
            config.method.batch_size = 4
            config.model.lstm_sequence_length = 3
            main(args, config)

    class TestModels:
        def test_no_lstm(self):
            args = parse_args()
            args.virtual_display = False
            args.debug_env = True
            args.wandb = False
            args.overrides.append('model=base')
            config = get_config(args)
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
