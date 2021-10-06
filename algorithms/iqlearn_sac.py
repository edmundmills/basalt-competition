from algorithms.sac import SoftActorCritic
from utils.datasets import MixedReplayBuffer
from algorithms.loss_functions.iqlearn import IQLearnLoss, IQLearnLossDRQ
from algorithms.loss_functions.sac import SACPolicyLoss
from utils.gpu import cat_batches


class IQLearnSAC(SoftActorCritic):
    def __init__(self, expert_dataset, config, **kwargs):
        super().__init__(config, pretraining=False, **kwargs)
        self.replay_buffer = MixedReplayBuffer(
            expert_dataset=expert_dataset,
            config=config,
            batch_size=config.method.batch_size,
            initial_replay_buffer=self.replay_buffer)

    def initialize_loss_functions(self):
        if self.drq:
            self._q_loss = IQLearnLossDRQ(self.online_q, self.config, self.target_q)
        else:
            self._q_loss = IQLearnLoss(self.online_q, self.config, self.target_q)
        self._policy_loss = SACPolicyLoss(self.actor, self.online_q, self.config)

    def update_q(self, expert_batch, replay_batch,
                 expert_batch_aug=None, replay_batch_aug=None):
        loss, metrics = self._q_loss(expert_batch, replay_batch,
                                     expert_batch_aug, replay_batch_aug)
        self.q_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.q_optimizer.step()
        return metrics

    def train_one_batch(self, step, batch):
        expert_batch, replay_batch = batch
        expert_batch = self.gpu_loader.expert_batch_to_device(expert_batch)
        replay_batch = self.gpu_loader.batch_to_device(replay_batch)
        expert_batch_aug = self.augmentation(expert_batch)
        replay_batch_aug = self.augmentation(replay_batch)
        combined_batch = cat_batches((expert_batch, replay_batch,
                                     expert_batch_aug, replay_batch_aug))
        if self.drq:
            q_metrics = self.update_q(expert_batch, replay_batch,
                                      expert_batch_aug, replay_batch_aug)
        else:
            q_metrics = self.update_q(expert_batch_aug, replay_batch_aug)
        policy_metrics = self.update_policy(step, combined_batch)

        metrics = {**policy_metrics, **q_metrics}
        return metrics
