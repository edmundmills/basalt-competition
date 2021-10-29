from algorithms.loss_functions.curious_iq import CuriousIQLoss
from algorithms.online_imitation import OnlineImitation
from core.state import cat_transitions


class CuriousIQ(OnlineImitation):
    def __init__(self, expert_dataset, agent, config, **kwargs):
        super().__init__(expert_dataset, agent, config, **kwargs)
        self.curiosity_pretraining_steps = config.method.curiosity_pretraining_steps
        self.loss_function = CuriousIQLoss(agent, config)
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
            metrics = self.train_one_batch(batch, curiosity_only=True)
            self.increment_step(metrics, profiler=None)

    def _update_curiosity(self, batch):
        curiosity_loss, metrics = self.curiosity_module.loss(batch)
        self.curiosity_optimizer.zero_grad(set_to_none=True)
        curiosity_loss.backward()
        self.curiosity_optimizer.step()
        return metrics

    def train_one_batch(self, batch, curiosity_only=False):
        (expert_batch, expert_idx), (replay_batch, replay_idx) = batch
        expert_batch = self.gpu_loader.transitions_to_device(expert_batch)
        replay_batch = self.gpu_loader.transitions_to_device(replay_batch)
        aug_expert_batch = self.augmentation(expert_batch)
        aug_replay_batch = self.augmentation(replay_batch)

        combined_batch = cat_transitions((expert_batch, replay_batch,
                                          aug_expert_batch, aug_replay_batch))
        curiosity_metrics = self._update_curiosity(combined_batch)

        if curiosity_only:
            return curiosity_metrics

        loss, metrics, final_hidden = self.loss_function(aug_expert_batch,
                                                         aug_replay_batch,
                                                         expert_batch,
                                                         replay_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if final_hidden.size()[0] != 0:
            final_hidden_expert, final_hidden_replay = final_hidden.chunk(2, dim=0)
            self.replay_buffer.update_hidden(replay_idx, final_hidden_replay,
                                             expert_idx, final_hidden_expert)

        return {**metrics, **curiosity_metrics}
