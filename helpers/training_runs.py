from pathlib import Path
import time
import os

import numpy as np
import matplotlib.pyplot as plt


class TrainingRun:
    def __init__(self,
                 config,
                 update_frequency=100,
                 checkpoint_freqency=None,
                 wandb=False):
        self.config = config
        self.wandb = wandb
        self.environment = config['environment']
        self.algorithm = config['algorithm']
        self.name = f'{self.environment}_{self.algorithm}_{int(round(time.time()))}'
        self.update_frequency = update_frequency
        self.checkpoint_freqency = checkpoint_freqency
        self.losses = []
        self.timestamps = []
        save_path = Path('training_runs') / self.name
        save_path.mkdir(exist_ok=True)
        self.save_path = save_path

    def step(self):
        self.timestamps.append(time.time())

    def print_update(self):
        iter_count = len(self.timestamps)
        if (iter_count % self.update_frequency) == 0 and len(self.timestamps) > 2:
            print(f'Iteration {iter_count} {self.iteration_rate():.2f} it/s')

    def iteration_rate(self):
        iterations = min(self.update_frequency, len(self.timestamps) - 1)
        duration = self.timestamps[-1] - self.timestamps[-iterations]
        rate = iterations / duration
        return rate

    def save_data(self):
        data = {'timestamps': self.timestamps,
                'config': self.config}
        np.save(file=self.save_path / 'data.npy', arr=np.array(data))
        # fig = plt.figure()
        # plt.ylabel('Loss')
        # plt.xlabel(f'iterations (x{self.update_frequency})')
        # plt.plot(self.smoothed_losses())
        # plt.savefig(self.save_path / 'loss_plot.png')
