from algorithms.loss_functions.iqlearn import IQLearnLoss, IQLearnLossDRQ
from algorithms.loss_functions.sqil import SQILLoss
from core.algorithm import Algorithm
from core.data_augmentation import DataAugmentation
from core.datasets import MixedReplayBuffer, MixedSequenceReplayBuffer
from core.state import update_hidden
from core.trajectories import TrajectoryGenerator
from modules.alpha_tuning import AlphaTuner
from modules.curriculum import CurriculumScheduler

import torch as th


class OnlineImitation(Algorithm):
    def __init__(self, expert_dataset, model, config,
                 initial_replay_buffer=None, initial_iter_count=0):
        super().__init__(config)
        # unpack config
        self.lr = config.method.learning_rate
        self.training_steps = config.method.training_steps
        self.batch_size = config.method.batch_size

        # misc training config
        self.drq = config.method.drq
        self.cyclic_learning_rate = config.cyclic_learning_rate

        self.model = model
        self.expert_dataset = expert_dataset
        self.initial_replay_buffer = initial_replay_buffer
        self.iter_count += initial_iter_count
        self.augmentation = DataAugmentation(config)
        self._initialize_loss_function(model, config)

        # modules
        self.alpha_tuner = AlphaTuner([self.model], config, self.context)

        self.curriculum_training = config.curriculum_training
        self.curriculum_scheduler = CurriculumScheduler(config) \
            if self.curriculum_training else None

        # context
        if config.context.name == 'MineRL':
            self.snowball_helper = self.context.snowball_helper
        else:
            self.snowball_helper = None

    def initialize_replay_buffer(self, expert_dataset, initial_replay_buffer=None):
        if initial_replay_buffer is not None:
            print((f'Using initial replay buffer'
                   f' with {len(initial_replay_buffer)} steps'))
        kwargs = dict(
            expert_dataset=expert_dataset,
            config=self.config,
            batch_size=self.batch_size,
            initial_replay_buffer=initial_replay_buffer
        )
        if self.config.lstm_layers == 0:
            replay_buffer = MixedReplayBuffer(**kwargs)
        else:
            replay_buffer = MixedSequenceReplayBuffer(**kwargs)
        self.replay_buffer = replay_buffer
        return replay_buffer

    def _initialize_loss_function(self, model, config):
        if config.method.loss_function == 'sqil':
            self.loss_function = SQILLoss(model, config)
        elif config.method.loss_function == 'iqlearn' and self.drq:
            self.loss_function = IQLearnLossDRQ(model, config)
        elif config.method.loss_function == 'iqlearn':
            self.loss_function = IQLearnLoss(model, config)

    def train_one_batch(self, batch):
        (expert_batch, expert_idx), (replay_batch, replay_idx) = batch
        expert_batch = self.gpu_loader.transitions_to_device(expert_batch)
        replay_batch = self.gpu_loader.transitions_to_device(replay_batch)
        aug_expert_batch = self.augmentation(expert_batch)
        aug_replay_batch = self.augmentation(replay_batch)

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

        return metrics

    def __call__(self, env, profiler=None):
        model = self.model
        expert_dataset = self.expert_dataset

        self.optimizer = th.optim.AdamW(model.parameters(), lr=self.lr)
        if self.cyclic_learning_rate:
            decay_factor = .25**(1/(self.training_steps/4))
            self.scheduler = th.optim.lr_scheduler.CyclicLR(self.optimizer,
                                                            base_lr=self.lr,
                                                            max_lr=self.lr*10,
                                                            mode='exp_range',
                                                            gamma=decay_factor,
                                                            step_size_up=2000,
                                                            cycle_momentum=False)

        replay_buffer = self.initialize_replay_buffer(expert_dataset,
                                                      self.initial_replay_buffer)

        self.trajectory_generator = TrajectoryGenerator(env, replay_buffer)
        self.trajectory_generator.start_new_trajectory()

        print((f'{self.algorithm_name}: Starting training'
               f' for {self.training_steps} steps (iteration {self.iter_count})'))

        for step in range(self.training_steps):
            metrics = {}
            current_state = replay_buffer.current_state()
            action, hidden = model.get_action(
                self.gpu_loader.state_to_device(current_state))

            suppressed_snowball = self.snowball_helper.suppressed_snowball(
                step, current_state, action) if self.snowball_helper else False

            if suppressed_snowball:
                next_state, reward, done, _ = env.step(-1)
            else:
                next_state, reward, done, _ = env.step(action)

            update_hidden(next_state, hidden)
            metrics['Rewards/ground_truth_reward'] = reward
            replay_buffer.append_step(action, reward, next_state, done)

            if self.curriculum_scheduler:
                metrics['Curriculum/inclusion_fraction'] = \
                    self.curriculum_scheduler.update_replay_buffer(self,
                                                                   replay_buffer, step)

            if self.alpha_tuner:
                self.alpha_tuner.update_model_alpha(step)
                metrics['alpha'] = model.alpha

            if len(replay_buffer) >= replay_buffer.replay_batch_size:
                batch = replay_buffer.sample(batch_size=self.batch_size)
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

            self.save_checkpoint(replay_buffer=replay_buffer, model=model)

            self.conditionally_increment_episode(step, replay_buffer.current_trajectory(),
                                                 suppressed_snowball=suppressed_snowball)

        print(f'{self.algorithm_name}: Training complete')
        return model, replay_buffer
