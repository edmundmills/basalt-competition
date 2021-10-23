from algorithms.algorithm import Algorithm
from core.environment import ObservationSpace, ActionSpace
from core.data_augmentation import DataAugmentation

import os
import wandb

import torch as th
from torch.core.data import DataLoader


class SupervisedLearning(Algorithm):
    def __init__(self, config, epochs=None):
        super().__init__(config)
        self.epochs = epochs if epochs else config.epochs
        self.lr = config.method.learning_rate
        self.batch_size = config.method.batch_size
        self.augmentation = DataAugmentation(config)
        self.gpu_loader.loading_sequences = False

    def __call__(self, model, train_dataset, test_dataset=None, _env=None):
        self.training_steps = len(train_dataset) * self.epochs / self.batch_size
        optimizer = th.optim.AdamW(model.parameters(), lr=self.lr)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=4)

        self.iter_count = 0
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
                    wandb.log({'TerminationCritic/train_loss': loss.detach()},
                              step=self.iter_count)

                self.iter_count += 1

            print(f'Epoch #{epoch + 1} completed')
            if test_dataset is not None and self.wandb:
                eval_metrics = self.eval(model, test_dataset)
                print('Metrics: ', eval_metrics)
                wandb.log(eval_metrics)

        print('Training complete')
        # model.save(os.path.join('train', f'{self.name}.pth'))
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
