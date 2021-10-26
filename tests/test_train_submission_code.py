from train_submission_code import *
from utility.config import get_config, parse_args

import copy


# def test_default_config(default_args, default_config):
#     main(default_args, default_config)


def test_sac():
    args = parse_args()
    args.virtual_display = False
    args.debug_env = True
    args.wandb = False
    args.overrides.append('method=iqlearn_sac')
    config = get_config(args)
    config.method.starting_steps = 100
    config.method.training_steps = 3
    config.method.batch_size = 4
    config.lstm_sequence_length = 3
    main(args, config)

#
# def test_bc():
#     args = parse_args()
#     args.virtual_display = False
#     args.debug_env = True
#     args.wandb = False
#     args.overrides.append('method=bc')
#     config = get_config(args)
#     config.method.max_training_steps = 3
#     config.method.batch_size = 4
#     config.lstm_sequence_length = 3
#     main(args, config)
#
#
# def test_iqlearn_offline():
#     args = parse_args()
#     args.virtual_display = False
#     args.debug_env = True
#     args.wandb = False
#     args.overrides.append('method=iqlearn_offline')
#     config = get_config(args)
#     config.method.max_training_steps = 3
#     config.method.batch_size = 4
#     config.lstm_sequence_length = 3
#     main(args, config)
#
#
# def test_no_lstm(default_args, default_config):
#     config = default_config
#     config.lstm_layers = 0
#     config.lstm_hidden_size = 0
#     main(default_args, config)


# def test_waterfall(default_args, default_config):
#     config = default_config
#     config.env.name = 'MineRLBasaltMakeWaterfall-v0'
#     main(default_args, config)
#
#
# def test_pen(default_args, default_config):
#     config = default_config
#     config.env.name = 'MineRLBasaltCreateVillageAnimalPen-v0'
#     main(default_args, config)
#
#
# def test_house(default_args, default_config):
#     config = default_config
#     config.env.name = 'MineRLBasaltBuildVillageHouse-v0'
#     main(default_args, config)
#
#
# def test_treechop(default_args, default_config):
#     config = default_config
#     config.env.name = 'MineRLTreechop-v0'
#     main(default_args, config)
#
#
# # def test_navigate(default_args, default_config):
# #     config = default_config
# #     config.env.name = 'MineRLNavigateDense-v0'
# #     main(default_args, config)
