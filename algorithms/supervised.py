from helpers.datasets import TestDataset
from helpers.environment import ObservationSpace, ActionSpace
from helpers.gpu import states_to_device
import os
import wandb

import torch as th
from torch.utils.data import DataLoader


class SupervisedLearning(Algorithm):
    def __init__(self, config):
        super().__init__(config)
        self.epochs = config['epochs']
        self.lr = config['learning_rate']
        self.batch_size = config['batch_size']

    def __call__(self, model, train_dataset, test_dataset=None):
        optimizer = th.optim.Adam(self.model.parameters(), lr=self.lr)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=4)
        if test_dataset is not None:
            test_dataset = TestDataset(test_dataset)

        iter_count = 0
        for epoch in range(self.epochs):
            for _, (dataset_obs, dataset_actions,
                    _next_obs, _done) in enumerate(train_dataloader):
                loss = model.loss(dataset_obs, dataset_actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_count += 1
                self.log_step()

            print(f'Epoch #{epoch + 1} completed')
            if self.test_dataset is not None and self.wandb:
                test_batch = test_dataset.sample()
                eval_metrics = self.eval(model, test_batch)
                print('Metrics: ', eval_metrics)
                wandb.log(
                    {**eval_metrics,
                     'average_its_per_s': self.iteration_rate()})

        print('Training complete')
        model.save(os.path.join('train', f'{self.name}.pth'))
        del dataloader

    def eval(self, model, batch):
        obs, actions, _next_obs, _done = batch
        with th.no_grad():
            test_loss = model.loss(obs, actions)
        eval_metrics = {'test_loss': test_loss.detach().item()}
        return eval_metrics
