from algorithms.algorithm import Algorithm
from helpers.environment import ObservationSpace, ActionSpace
from helpers.gpu import states_to_device
import os
import wandb

import torch as th
from torch.utils.data import DataLoader


class SupervisedLearning(Algorithm):
    def __init__(self, config):
        super().__init__(config)
        self.epochs = config.method.epochs
        self.lr = config.method.learning_rate
        self.batch_size = config.method.batch_size

    def __call__(self, model, train_dataset, test_dataset=None, _env=None):
        optimizer = th.optim.Adam(model.parameters(), lr=self.lr)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=4)

        iter_count = 0
        for epoch in range(self.epochs):
            for _, (dataset_obs, dataset_actions,
                    _next_obs, _done) in enumerate(train_dataloader):
                loss = model.loss(dataset_obs, dataset_actions)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.wandb:
                    wandb.log({'train_loss': loss.detach()}, step=self.iter_count)

                iter_count += 1
                self.log_step()

            print(f'Epoch #{epoch + 1} completed')
            if test_dataset is not None and self.wandb:
                eval_metrics = self.eval(model, test_dataset)
                print('Metrics: ', eval_metrics)
                wandb.log(eval_metrics)

        print('Training complete')
        model.save(os.path.join('train', f'{self.name}.pth'))
        return model, None

    def eval(self, model, test_dataset):
        test_losses = []
        dataloader = DataLoader(test_dataset,
                                shuffle=False,
                                batch_size=self.batch_size,
                                num_workers=4,
                                drop_last=True)
        for obs, actions, _next_obs, _done in dataloader:
            with th.no_grad():
                test_loss = model.loss(obs, actions)
                test_losses.append(test_loss.detach().item())
        test_loss = sum(test_losses) / len(test_losses)
        eval_metrics = {'test_loss': test_loss}
        return eval_metrics
