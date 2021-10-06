from algorithms.algorithm import Algorithm
from utils.environment import ObservationSpace, ActionSpace
from utils.data_augmentation import DataAugmentation

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
        self.augmentation = DataAugmentation(config)

    def __call__(self, model, train_dataset, test_dataset=None, _env=None):
        optimizer = th.optim.Adam(model.parameters(), lr=self.lr)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=4)

        iter_count = 0
        for epoch in range(self.epochs):
            for batch in train_dataloader:
                batch = self.gpu_loader.expert_batch_to_device(batch)
                batch = self.augmentation(batch)
                states, actions, _next_states, _done, _rewards = batch
                loss = model.loss(states, actions)

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
        for batch in dataloader:
            batch = expert_batch_to_device(batch)
            states, actions, _next_states, _done, _rewards = batch
            with th.no_grad():
                test_loss = model.loss(states, actions)
                test_losses.append(test_loss.detach().item())
        test_loss = sum(test_losses) / len(test_losses)
        eval_metrics = {'test_loss': test_loss}
        return eval_metrics
