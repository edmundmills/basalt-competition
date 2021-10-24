

class CurriculumScheduler:
    def __init__(self, config):
        self.initial_curriculum_size = config.initial_curriculum_size
        self.curriculum_fraction_of_training = config.env.curriculum_fraction_of_training
        self.variable_training_episode_length = config.variable_training_episode_length

    def curriculum_fraction(self, algorithm, step):
        if not algorithm.curriculum_training:
            return 1
        curriculum_fraction = step / (self.curriculum_fraction_of_training
                                      * algorithm.training_steps)
        return curriculum_fraction

    def update_replay_buffer(self, algorithm, replay_buffer, step):
        curriculum_fraction = self.curriculum_fraction(algorithm, step)
        curriculum_inclusion = replay_buffer.update_curriculum(step, curriculum_fraction)
        return curriculum_inclusion

    def max_episode_length(self, algorithm, step):
        curriculum_fraction = self.curriculum_fraction(algorithm, step)
        curriculum_span = algorithm.max_training_episode_length \
            - self.initial_curriculum_size
        episode_length = int(self.initial_curriculum_size +
                             curriculum_fraction * curriculum_span)
        return min(algorithm.max_training_episode_length,
                   max(episode_length, algorithm.min_training_episode_length))
