from modules.curriculum import CurriculumScheduler
from modules.alpha_tuning import AlphaTuner
from core.trajectories import TrajectoryGenerator
from core.state import update_hidden
from core.datasets import MixedReplayBuffer, MixedSequenceReplayBuffer
from algorithms.loss_functions.iqlearn import IQLearnLoss, IQLearnLossDRQ
from core.algorithm import Algorithm
from core.data_augmentation import DataAugmentation

import os
import wandb

import torch as th
from torch.utils.data import DataLoader


class SupervisedLearning(Algorithm):
    def __init__(self, model, train_dataset, config, test_dataset=None):
        super().__init__(config)
        # unpack config
        self.lr = config.method.learning_rate
        self.epochs = config.method.epochs
        self.batch_size = config.method.batch_size
        self.training_steps = len(train_dataset) * self.epochs / self.batch_size

        # misc training config
        self.drq = config.method.drq
        self.cyclic_learning_rate = config.cyclic_learning_rate

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.augmentation = DataAugmentation(config)
        self._initialize_loss_function(model, config)

        # modules
        self.alpha_tuner = AlphaTuner([self.model], config, self.context)

        self.curriculum_training = config.curriculum_training
        self.curriculum_scheduler = CurriculumScheduler(config) \
            if self.curriculum_training else None

    def _initialize_loss_function(self, model, config):
        if config.method.loss_function == 'bc':
            self.loss_function = BCLoss(model, config)
        elif config.method.loss_function == 'sqil':
            self.loss_function = SqilLoss(model, config)
        elif config.method.loss_function == 'iqlearn' and self.drq:
            self.loss_function = IQLearnLossDRQ(model, config)
        elif config.method.loss_function == 'iqlearn':
            self.loss_function = IQLearnLoss(model, config)

    def train_one_batch(self, batch):
        expert_batch, expert_idx = batch
        expert_batch = self.gpu_loader.transitions_to_device(expert_batch)
        aug_expert_batch = self.augmentation(expert_batch)

        loss, metrics, final_hidden = self.loss_function(expert=aug_expert_batch,
                                                         expert_no_aug=expert_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if final_hidden.size()[0] != 0:
            self.train_dataset.update_hidden(expert_idx, final_hidden_expert)

        return metrics

    def validation_metrics(self, model, test_dataset):
        if test_dataset is not None and self.wandb:
            test_losses = []
            dataloader = DataLoader(test_dataset,
                                    shuffle=False,
                                    batch_size=self.batch_size,
                                    num_workers=4,
                                    drop_last=True)
            for batch in dataloader:
                batch = self.gpu_loader.batch_to_device(batch)
                with th.no_grad():
                    validation_loss, _metrics, _final_hidden = self.loss_function(batch)
                    validation_losses.append(validation_loss.detach().item())
            validation_loss = sum(validation_losses) / len(validation_losses)
            validation_metrics = {'Validation/loss': validation_loss}
            print('Metrics: ', validation_metrics)
            wandb.log(validation_metrics, step=self.iter_count)

    def __call__(self, _env=None, profiler=None):
        train_dataset = self.train_dataset
        test_dataset = self.test_dataset
        model = self.model
        print((f'{self.algorithm_name}: Starting training'
               f' for {self.training_steps} steps (iteration {self.iter_count})'))

        self.optimizer = th.optim.AdamW(model.parameters(), lr=self.lr)
        if self.cyclic_learning_rate:
            decay_factor = .25**(1/(self.training_steps/4))
            self.scheduler = th.optim.lr_scheduler.CyclicLR(self.optimizer,
                                                            base_lr=self.lr,
                                                            max_lr=self.lr*10,
                                                            mode='exp_range',
                                                            gamma=decay_factor,
                                                            step_size_up=self.epochs/2,
                                                            cycle_momentum=False)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=4)

        self.iter_count = 0
        for epoch in range(self.epochs):
            for batch in train_dataloader:
                metrics = {}

                if self.curriculum_scheduler:
                    metrics['Curriculum/inclusion_fraction'] = \
                        self.curriculum_scheduler.update_replay_buffer(self,
                                                                       replay_buffer,
                                                                       step)

                training_metrics = self.train_one_batch(batch)
                metrics = {**metrics, **training_metrics}

                if self.cyclic_learning_rate:
                    self.scheduler.step()
                    metrics['learning_rate'] = self.scheduler.get_last_lr()[0]

                if self.alpha_tuner and self.alpha_tuner.entropy_tuning:
                    alpha_metrics = self.alpha_tuner.update_alpha(metrics['entropy'])
                    metrics = {**metrics, **alpha_metrics}

                self.log_step(metrics, profiler)

                if self.shutdown_time_reached():
                    break

                self.iter_count += 1

                self.save_checkpoint(model=model)

            print(f'Epoch #{epoch + 1} completed')
            self.validation_metrics(model, test_dataset)

        print(f'{self.algorithm_name}: Training complete')
        return model, None
