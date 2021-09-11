from pathlib import Path
import time
import os

import numpy as np
import matplotlib.pyplot as plt


class TrainingRun:
    def __init__(self,
                 label,
                 lr,
                 epochs=None,
                 training_steps=None,
                 update_frequency=100,
                 checkpoint_freqency=None):
        self.environment = os.getenv('MINERL_ENVIRONMENT')
        self.name = f'{self.environment}_{label}_{int(round(time.time()))}'
        self.lr = lr
        self.epochs = epochs
        self.training_steps = training_steps
        self.update_frequency = update_frequency
        self.checkpoint_freqency = checkpoint_freqency
        self.losses = []
        self.timestamps = []
        save_path = Path('training_runs') / self.name
        save_path.mkdir(exist_ok=True)
        self.save_path = save_path

    def append_loss(self, loss):
        self.losses.append(loss)
        self.timestamps.append(time.time())

    def print_update(self, iter_count):
        if iter_count == 1:
            print('Training Starting')
        elif (iter_count % self.update_frequency) == 0:
            if len(self.losses) == 0:
                return
            smoothed_loss = sum(
                self.losses[-self.update_frequency:-1])/self.update_frequency
            iterations = min(self.update_frequency, len(self.timestamps) - 1)
            duration = self.timestamps[-1] - self.timestamps[-iterations]
            rate = iterations / duration
            print(
                f'Iteration {iter_count}. Loss: {smoothed_loss:.2f}, {rate:.2f} it/s')

    def smoothed_losses(self):
        smoothed_losses = []
        current_bucket = []
        for loss in self.losses:
            current_bucket.append(loss)
            if len(current_bucket) == self.update_frequency:
                smoothed_losses.append(sum(current_bucket)/len(current_bucket))
                current_bucket.clear()
        return smoothed_losses

    def save_data(self):
        data = {'timestamps': self.timestamps,
                'losses': self.losses,
                'lr': self.lr,
                'epochs': self.epochs,
                'training_steps': self.training_steps,
                }
        np.save(file=self.save_path / 'data.npy', arr=np.array(data))
        fig = plt.figure()
        plt.ylabel('Loss')
        plt.xlabel(f'iterations (x{self.update_frequency})')
        plt.plot(self.smoothed_losses())
        plt.savefig(self.save_path / 'loss_plot.png')
