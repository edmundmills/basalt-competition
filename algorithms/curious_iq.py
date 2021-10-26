from algorithms.online_imitation import OnlineImitation


class CuriousIQ(OnlineImitation):
    def __init__(self, expert_dataset, config, **kwargs):
        super().__init__(config, pretraining=False, **kwargs)
        self.online_curiosity_training = config.method.online_curiosity_training
        self.initial_curiosity_fraction = config.method.initial_curiosity_fraction
        self.iqlearn_q = SoftQAgent(config).to(self.device)
        if config.method.target_q:
            self.iqlearn_target_q = SoftQAgent(config).to(self.device)
            self.iqlearn_target_q.load_state_dict(self.iqlearn_q.state_dict())
            disable_gradients(self.iqlearn_target_q)
            print('IQLearn Target Network Initialized')
        else:
            self.iqlearn_target_q = None
        if self.drq:
            self._iqlearn_loss = IQLearnLossDRQ(self.iqlearn_q, config,
                                                target_q=self.iqlearn_target_q)
        else:
            self._iqlearn_loss = IQLearnLoss(self.iqlearn_q, config,
                                             target_q=self.iqlearn_target_q)
        self.iqlearn_optimizer = th.optim.Adam(self.iqlearn_q.parameters(),
                                               lr=config.method.iqlearn_lr)
        self._policy_loss = CuriousIQPolicyLoss(self.actor, self.online_q, self.iqlearn_q,
                                                config)
        if self.entropy_tuning:
            self.initialize_alpha_optimization()

        kwargs = dict(
            expert_dataset=expert_dataset,
            config=config,
            batch_size=config.method.batch_size,
            initial_replay_buffer=self.replay_buffer)
        if self.config.lstm_layers == 0:
            self.replay_buffer = MixedReplayBuffer(**kwargs)
        else:
            self.replay_buffer = MixedSequenceReplayBuffer(**kwargs)

    def update_model_alphas(self):
        alpha = self._policy_loss.log_alpha.detach().exp().item()
        self.online_q.alpha = alpha
        self.target_q.alpha = alpha
        self.actor.alpha = alpha
        self.iqlearn_q.alpha = alpha
        self.iqlearn_target_q.alpha = alpha

    def _soft_update_target(self):
        for target, online in zip(
                self.target_q.parameters(), self.online_q.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + online.data * self.tau)
        if self.iqlearn_target_q is None:
            return
        for target, online in zip(self.iqlearn_target_q.parameters(),
                                  self.iqlearn_q.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + online.data * self.tau)

    def update_iqlearn(self, expert_batch, replay_batch,
                       expert_batch_aug=None, replay_batch_aug=None):
        loss, metrics = self._iqlearn_loss(expert_batch, replay_batch,
                                           expert_batch_aug, replay_batch_aug)
        self.iqlearn_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.iqlearn_optimizer.step()
        return metrics

    def train_one_batch(self, step, batch, curiosity_only=False):
        (expert_batch, expert_idx), (replay_batch, replay_idx) = batch
        expert_batch = self.gpu_loader.expert_batch_to_device(expert_batch)
        replay_batch = self.gpu_loader.batch_to_device(replay_batch)
        expert_batch_aug = self.augmentation(expert_batch)
        replay_batch_aug = self.augmentation(replay_batch)
        combined_batch = cat_batches((expert_batch, replay_batch,
                                     expert_batch_aug, replay_batch_aug))
        if not curiosity_only:
            if self.drq:
                q_metrics = self.update_q(replay_batch, replay_batch_aug)
                iqlearn_metrics = self.update_iqlearn(expert_batch, replay_batch,
                                                      expert_batch_aug, replay_batch_aug)
            else:
                q_metrics = self.update_q(replay_batch_aug)
                iqlearn_metrics = self.update_iqlearn(expert_batch_aug, replay_batch_aug)
            policy_metrics = self.update_policy(step, combined_batch)
        else:
            q_metrics = {}
            iqlearn_metrics = {}
            policy_metrics = {}
        if curiosity_only or \
                (self.online_curiosity_training and
                 step < (self.config.method.curiosity_only_steps
                         + self.config.method.curiosity_fade_out_steps)):
            curiosity_metrics = self.update_curiosity(combined_batch)
        else:
            curiosity_metrics = {}

        metrics = {**policy_metrics, **q_metrics, **curiosity_metrics, **iqlearn_metrics}
        return metrics
