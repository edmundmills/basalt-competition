import random


class CurriculumScheduler:
    def __init__(self, config):
        self.current_curriculum_length = 0
        self.complete = False

        self.initial_curriculum_size = config.dataset.initial_curriculum_size
        self.curriculum_fraction_of_training = \
            config.dataset.curriculum_fraction_of_training
        self.final_curriculum_fraction = 1
        self.curriculum_refresh_steps = config.dataset.curriculum_refresh_steps
        self.variable_training_episode_length = \
            config.dataset.variable_training_episode_length
        self.extracurricular_sparsity = config.dataset.extracurricular_sparsity

        # emphasize recent samples
        self.emphasize_new_samples = config.dataset.emphasize_new_samples
        self.emphasized_fraction = config.dataset.emphasized_fraction
        self.emphasis_relative_sample_frequency = \
            config.dataset.emphasis_relative_sample_frequency
        if self.emphasize_new_samples:
            self.final_curriculum_fraction += self.emphasized_fraction

    def curriculum_fraction(self, algorithm, step):
        if not algorithm.curriculum_training:
            return 1
        curriculum_fraction = step / (self.curriculum_fraction_of_training
                                      * algorithm.training_steps)
        return curriculum_fraction

    def max_episode_length(self, algorithm, step):
        curriculum_fraction = self.curriculum_fraction(algorithm, step)
        curriculum_span = algorithm.max_training_episode_length \
            - self.initial_curriculum_size
        episode_length = int(self.initial_curriculum_size +
                             curriculum_fraction * curriculum_span)
        return min(algorithm.max_training_episode_length,
                   max(episode_length, algorithm.min_training_episode_length))

    def update_expert_dataset(self, dataset, curriculum_fraction):
        random_seed = random.randint(0, self.extracurricular_sparsity - 1)
        filtered_lookup, master_indices = \
            zip(*[[(t_idx, sequence_idx), master_idx]
                  for master_idx, (t_idx, sequence_idx)
                  in enumerate(dataset.master_lookup)
                  if (sequence_idx <= (len(dataset.trajectories[t_idx])
                                       * curriculum_fraction)
                      or sequence_idx < self.initial_curriculum_size
                      or (sequence_idx+random_seed) % self.extracurricular_sparsity == 0)
                  ])
        filtered_lookup = list(filtered_lookup)
        master_indices = list(master_indices)
        self.current_curriculum_length = len(filtered_lookup)

        # emphasize recently added samples
        if self.emphasize_new_samples and curriculum_fraction > self.emphasized_fraction \
                and curriculum_fraction < self.final_curriculum_fraction:
            for i in range(self.emphasis_relative_sample_frequency - 1):
                emphasis_lookup, emphasis_master_indices = \
                    zip(*[[(t_idx, sequence_idx), master_idx]
                          for master_idx, (t_idx, sequence_idx)
                          in enumerate(dataset.master_lookup)
                          if (sequence_idx <=
                              (len(dataset.trajectories[t_idx])
                               * curriculum_fraction)
                              and sequence_idx > (len(dataset.trajectories[t_idx])
                                                  * (curriculum_fraction
                                                     - self.emphasized_fraction)))])
                filtered_lookup.extend(emphasis_lookup)
                master_indices.extend(emphasis_master_indices)
                print(f'{len(emphasis_lookup)} samples emphasized')

        dataset.active_lookup = filtered_lookup
        dataset.cross_lookup = {filtered_idx: master_idx
                                for filtered_idx, master_idx in enumerate(master_indices)}
        print(f'Expert curriculum updated, including {self.current_curriculum_length}'
              f' / {len(dataset.master_lookup)} sequences')

    def update_replay_buffer(self, algorithm, replay_buffer, step):
        expert_dataset = replay_buffer.expert_dataset
        curriculum_fraction = self.curriculum_fraction(algorithm, step)
        current_curriculum_inclusion = self.current_curriculum_length / \
            len(expert_dataset.master_lookup)
        if self.current_curriculum_length == 0 \
                or (step % self.curriculum_refresh_steps == 0 and not self.complete):
            self.update_expert_dataset(expert_dataset, curriculum_fraction)
            replay_buffer.expert_dataloader = replay_buffer._initialize_dataloader()
            if curriculum_fraction >= self.final_curriculum_fraction:
                self.complete = True
        curriculum_inclusion = self.current_curriculum_length \
            / len(expert_dataset.master_lookup)
        return curriculum_inclusion
