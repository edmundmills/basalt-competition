from train_submission_code import *

import copy


def test_default_config(default_args, default_config):
    main(default_args, default_config)


def test_no_lstm(default_args, default_config):
    config = default_config
    config.lstm_layers = 0
    config.lstm_hidden_size = 0
    main(default_args, config)
