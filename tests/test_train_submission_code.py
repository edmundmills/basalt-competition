from train_submission_code import *

import copy


def test_default_config(default_args, default_config):
    main(default_args, default_config)


def test_no_lstm(default_args, default_config):
    config = default_config
    config.lstm_layers = 0
    config.lstm_hidden_size = 0
    main(default_args, config)


def test_waterfall(default_args, default_config):
    config = default_config
    config.env.name = 'MineRLBasaltMakeWaterfall-v0'
    main(default_args, config)


def test_pen(default_args, default_config):
    config = default_config
    config.env.name = 'MineRLBasaltCreateVillageAnimalPen-v0'
    main(default_args, config)


def test_house(default_args, default_config):
    config = default_config
    config.env.name = 'MineRLBasaltBuildVillageHouse-v0'
    main(default_args, config)


def test_treechop(default_args, default_config):
    config = default_config
    config.env.name = 'MineRLTreechop-v0'
    main(default_args, config)


# def test_navigate(default_args, default_config):
#     config = default_config
#     config.env.name = 'MineRLNavigateDense-v0'
#     main(default_args, config)
