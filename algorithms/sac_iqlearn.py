from algorithms.sac import SoftActorCritic
from algorithms.loss_functions.iqlearn import IQLearnLoss
from algorithms.loss_functions.sac import SACPolicyLoss
from core.datasets import MixedReplayBuffer, MixedSequenceReplayBuffer
from core.state import cat_transitions
from modules.curriculum import CurriculumScheduler


class IQLearnSAC(SoftActorCritic):
    def __init__(self, expert_dataset, config, **kwargs):
        super().__init__(config, pretraining=False, **kwargs)
        kwargs = dict(
            expert_dataset=expert_dataset,
            config=config,
            batch_size=config.method.batch_size,
            initial_replay_buffer=self.replay_buffer)
        if self.config.model.lstm_layers == 0:
            self.replay_buffer = MixedReplayBuffer(**kwargs)
        else:
            self.replay_buffer = MixedSequenceReplayBuffer(**kwargs)

        if config.method.entropy_tuning and config.method.match_expert_entropy:
            self.alpha_tuner.target_entropy = expert_dataset.expert_policy_entropy

        self.curriculum_training = config.dataset.curriculum_training
        self.curriculum_scheduler = CurriculumScheduler(config) \
            if self.curriculum_training else None

    def _initialize_loss_functions(self):
        self._q_loss = IQLearnLoss(self.online_q, self.config, self.target_q)
        self._policy_loss = SACPolicyLoss(self.agent, self.online_q, self.config)

    def pre_train_step_modules(self, step):
        metrics = {}
        if self.curriculum_scheduler:
            metrics['Curriculum/inclusion_fraction'] = \
                self.curriculum_scheduler.update_replay_buffer(self,
                                                               self.replay_buffer, step)

        if self.alpha_tuner and self.alpha_tuner.decay_alpha:
            self.alpha_tuner.update_model_alpha(step)
            metrics['alpha'] = self.agent.alpha
        return metrics

    def _update_q(self, expert_batch, replay_batch,
                  expert_batch_aug=None, replay_batch_aug=None):
        loss, metrics, _ = self._q_loss(expert=expert_batch_aug,
                                        policy=replay_batch_aug,
                                        expert_aug=expert_batch,
                                        policy_aug=replay_batch)
        self.q_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.q_optimizer.step()
        return metrics

    def train_one_batch(self, batch):
        (expert_batch, expert_idx), (replay_batch, replay_idx) = batch
        expert_batch = self.gpu_loader.transitions_to_device(expert_batch)
        replay_batch = self.gpu_loader.transitions_to_device(replay_batch)
        expert_batch_aug = self.augmentation(expert_batch)
        replay_batch_aug = self.augmentation(replay_batch)
        combined_batch = cat_transitions((expert_batch, replay_batch,
                                          expert_batch_aug, replay_batch_aug))

        q_metrics = self._update_q(expert_batch_aug, replay_batch_aug,
                                   expert_batch, replay_batch)
        policy_metrics, final_hidden = self._update_policy(combined_batch)

        metrics = {**policy_metrics, **q_metrics}

        if final_hidden.size()[0] != 0:
            final_hidden_expert, final_hidden_replay, _, _ = final_hidden.chunk(4, dim=0)
            self.replay_buffer.update_hidden(replay_idx, final_hidden_replay,
                                             expert_idx, final_hidden_expert)

        if self.alpha_tuner and self.alpha_tuner.entropy_tuning:
            alpha_metrics = self.alpha_tuner.update_alpha(metrics['entropy'])
            metrics = {**metrics, **alpha_metrics}

        return metrics
