from algorithms.algorithm import Algorithm
from algorithms.loss_functions.iqlearn import IQLearnLoss
from algorithms.loss_functions.sqil import SqilLoss
from helpers.environment import ObservationSpace, ActionSpace
from helpers.datasets import MixedReplayBuffer

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
import os


class OfflineImitation(Algorithm):
    def __init__(self, expert_dataset, model, config, termination_critic=None):
        super().__init__(config)
        self.termination_critic = termination_critic
        self.lr = config.learning_rate
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.model = model
        self.expert_dataset = expert_dataset

    def __call__(self, _env=None, profiler=None):
        model = self.model
        expert_dataset = self.expert_dataset
        dataloader = DataLoader(expert_dataset, num_workers=4,
                                shuffle=True, batch_size=self.batch_size)

        if self.config.loss_function == 'sqil':
            self.loss_function = SqilLoss(model, self.config)
        elif self.config.loss_function == 'iqlearn':
            self.loss_function = IQLearnLoss(model, self.config)

        optimizer = th.optim.Adam(model.parameters(),
                                  lr=self.lr)

        for epoch in range(self.epochs):
            for step, sample in enumerate(dataloader):
                iter_count = step + 1
                loss, metrics = self.loss_function(*sample)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.wandb:
                    wandb.log(metrics, step=step)

            self.log_step()

            if self.checkpoint_freqency and \
                iter_count % self.checkpoint_freqency == 0 \
                    and iter_count < self.training_steps:
                self.save_checkpoint(iter_count,
                                     models_with_names=((model, 'model')))

            if profiler:
                profiler.step()

        print('Training complete')
        return model, None
