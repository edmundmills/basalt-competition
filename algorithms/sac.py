from agents.soft_q import SoftQAgent, TwinnedSoftQAgent
from algorithms.loss_functions.sac import SACQLoss, SACPolicyLoss
from algorithms.online import OnlineTraining
from core.networks import disable_gradients
from modules.alpha_tuning import AlphaTuner
from modules.curriculum import CurriculumScheduler

import torch as th


class SoftActorCritic(OnlineTraining):
    def __init__(self, config, agent=None, **kwargs):
        super().__init__(config, **kwargs)
        self.drq = config.method.drq
        self.cyclic_learning_rate = config.cyclic_learning_rate
        self.tau = config.method.tau
        self.double_q = config.method.double_q
        self.target_update_interval = config.method.target_update_interval

        # Set up networks - actor
        if agent is not None:
            self.agent = agent.to(self.device)
        else:
            self.agent = SoftQAgent(config).to(self.device)

        # Set up networks - critic
        if self.double_q:
            self.online_q = TwinnedSoftQAgent(config).to(self.device)
            self.target_q = TwinnedSoftQAgent(config).to(self.device)
        else:
            self.online_q = SoftQAgent(config).to(self.device)
            self.target_q = SoftQAgent(config).to(self.device)
        self.target_q.load_state_dict(self.online_q.state_dict())
        disable_gradients(self.target_q)

        # Loss functions
        self._initialize_loss_functions()

        # Optimizers
        self.q_optimizer = th.optim.Adam(self.online_q.parameters(),
                                         lr=config.method.q_lr)
        self.policy_optimizer = th.optim.Adam(self.agent.parameters(),
                                              lr=config.method.policy_lr)

        if self.cyclic_learning_rate:
            decay_factor = .25**(1/(self.training_steps/4))
            self.scheduler = th.optim.lr_scheduler.CyclicLR(self.q_optimizer,
                                                            base_lr=config.method.q_lr,
                                                            max_lr=config.method.q_lr*10,
                                                            mode='exp_range',
                                                            gamma=decay_factor,
                                                            step_size_up=2000,
                                                            cycle_momentum=False)

        # Modules
        if config.method.entropy_tuning:
            target_entropy = AlphaTuner.target_entropy(
                self.context, config.method.target_entropy_ratio)
        else:
            target_entropy = None
        self.alpha_tuner = AlphaTuner([self.agent, self.online_q, self.target_q], config,
                                      target_entropy=target_entropy)

    def _initialize_loss_functions(self):
        self._q_loss = SACQLoss(self.online_q, self.target_q, self.config)
        self._policy_loss = SACPolicyLoss(self.agent, self.online_q, self.config)

    def pre_train_step_modules(self, step):
        metrics = {}
        if self.alpha_tuner and self.alpha_tuner.decay_alpha:
            self.alpha_tuner.update_model_alpha(step)
            metrics['alpha'] = self.agent.alpha
        return metrics

    def _update_q(self, batch, batch_aug=None):
        q_loss, metrics = self._q_loss(batch, batch_aug=batch_aug)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        return metrics

    def _update_policy(self, batch):
        policy_loss, final_hidden, metrics = self._policy_loss(batch)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()
        return metrics, final_hidden

    def train_one_batch(self, batch):
        batch, batch_idx = batch
        batch = self.gpu_loader.transitions_to_device(batch)
        aug_batch = self.augmentation(batch)

        q_metrics = self._update_q(aug_batch, batch_aug=batch)
        policy_metrics, final_hidden = self._update_policy(aug_batch)

        metrics = {**policy_metrics, **q_metrics}

        if final_hidden.size()[0] != 0:
            self.replay_buffer.update_hidden(batch_idx, final_hidden)

        if self.alpha_tuner and self.alpha_tuner.entropy_tuning:
            alpha_metrics = self.alpha_tuner.update_alpha(metrics['entropy'])
            metrics = {**metrics, **alpha_metrics}

        return metrics

    def _soft_update_target(self):
        for target, online in zip(self.target_q.parameters(), self.online_q.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + online.data * self.tau)

    def post_train_step_modules(self, step):
        metrics = {}
        if self.cyclic_learning_rate:
            self.scheduler.step()
            metrics['learning_rate'] = self.scheduler.get_last_lr()[0]

        if step % self.target_update_interval:
            self._soft_update_target()
        return metrics
