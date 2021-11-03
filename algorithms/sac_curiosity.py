from algorithms.sac import SoftActorCritic
from modules.intrinsic_curiosity import CuriosityModule

import torch as th


class CuriositySAC(SoftActorCritic):
    def __init__(self, config, actor=None, **kwargs):
        super().__init__(config, actor, **kwargs)
        self.curiosity_pretraining_steps = config.method.curiosity_pretraining_steps
        self.curiosity_module = CuriosityModule(config).to(self.device)
        self.curiosity_optimizer = th.optim.Adam(self.curiosity_module.parameters(),
                                                 lr=config.method.curiosity_lr)

    def pretraining_modules(self):
        if self.curiosity_pretraining_steps > 0:
            self._train_curiosity_module()

    def _train_curiosity_module(self):
        print((f'Pretraining curiosity module for'
              f' {self.curiosity_pretraining_steps} steps'))
        for step in range(self.curiosity_pretraining_steps):
            batch = self.replay_buffer.sample(batch_size=self.batch_size)
            metrics = self.train_one_batch(step, batch, curiosity_only=True)
            self.increment_step(metrics, profiler=None)

    def _update_curiosity(self, batch):
        curiosity_loss, metrics = self.curiosity_module.loss(batch)
        self.curiosity_optimizer.zero_grad(set_to_none=True)
        curiosity_loss.backward()
        self.curiosity_optimizer.step()
        return metrics

    def train_one_batch(self, batch, curiosity_only=False):
        batch, batch_idx = batch
        batch = self.gpu_loader.batch_to_device(batch)
        with th.no_grad():
            batch = self.curiosity_module.rewards(batch, return_transition=True)
        aug_batch = self.augmentation(batch)
        if not curiosity_only:
            q_metrics = self._update_q(aug_batch, batch_aug=batch)
            policy_metrics, final_hidden = self._update_policy(aug_batch)
        else:
            policy_metrics = {}
            q_metrics = {}
        curiosity_metrics = self._update_curiosity(aug_batch)

        if final_hidden.size()[0] != 0:
            self.replay_buffer.update_hidden(batch_idx, final_hidden)

        metrics = {**policy_metrics, **q_metrics, **curiosity_metrics}
        return metrics
