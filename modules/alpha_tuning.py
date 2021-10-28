
import numpy as np
import torch as th


class AlphaTuner:
    def __init__(self, models, config, target_entropy=None):
        self.models = models
        self.target_entropy = target_entropy
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.decay_alpha = config.method.decay_alpha
        self.entropy_tuning = config.method.entropy_tuning
        self.initial_alpha = config.method.alpha
        if self.decay_alpha:
            self._initialize_alpha_decay(config)
        elif self.entropy_tuning:
            self._initialize_alpha_optimization(config)

    def _initialize_alpha_decay(self, config):
        self.final_alpha = config.method.final_alpha
        self.decay_steps = config.method.training_steps

    def _initialize_alpha_optimization(self, config):
        self.entropy_lr = config.method.entropy_lr
        print('Target entropy: ', self.target_entropy)
        self.log_alpha = th.tensor(np.log(self.initial_alpha),
                                   device=self.device, requires_grad=True)
        self.optimizer = th.optim.Adam([self.log_alpha], lr=self.entropy_lr)

    def current_alpha(self, step=0):
        if self.entropy_tuning:
            alpha = self.log_alpha.detach().exp()
        elif self.decay_alpha:
            alpha = max(self.initial_alpha - ((step / self.decay_steps)
                                              * (self.initial_alpha - self.final_alpha)),
                        self.final_alpha)
        else:
            alpha = self.initial_alpha
        return alpha

    def update_model_alpha(self, step=0):
        for model in self.models:
            model.alpha = self.current_alpha(step)

    def update_alpha(self, entropy):
        loss = (-self.log_alpha.exp() * (self.target_entropy - entropy)).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.update_model_alpha()
        metrics = {'alpha': self.current_alpha(), 'alpha_loss': loss.detach()}
        return metrics

    def target_entropy(context, target_entropy_ratio):
        target_entropy = (-np.log(1.0 / len(context.actions)) * target_entropy_ratio)
        return target_entropy
