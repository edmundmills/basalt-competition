from contexts.minerl.environment import MineRLContext
from core.gpu import GPULoader
from core.state import update_hidden
from core.trajectories import Trajectory

import numpy as np


class TrajectoryGenerator:
    def __init__(self, env, agent, config, replay_buffer=None, training=False):
        self.env = env
        self.agent = agent
        self.config = config
        self.replay_buffer = replay_buffer
        self.gpu_loader = GPULoader(config)
        if config.context.name == 'MineRL':
            self.context = MineRLContext(config)
            self.termination_helper = self.context.termination_helper
        self.training = training

    def new_trajectory(env, replay_buffer, reset_env=True):
        if len(replay_buffer.current_trajectory()) > 0:
            current_state = replay_buffer.current_state()
            replay_buffer.new_trajectory()
        else:
            current_state = None
        state = env.reset() if reset_env or current_state is None else current_state
        replay_buffer.current_trajectory().states.append(state)
        return replay_buffer.current_state()

    def start_new_trajectory(self, **kwargs):
        current_state = TrajectoryGenerator.new_trajectory(
            self.env, self.replay_buffer, **kwargs)
        return current_state

    def random_action(self):
        action = np.random.choice(self.context.actions)
        return action

    def env_interaction_step(self, step, trajectory=None, random_action=False):
        metrics = {}
        current_state = self.replay_buffer.current_state() \
            if trajectory is None else trajectory.current_state()
        if random_action:
            action = self.random_action()
            hidden = self.context.initial_hidden
        else:
            action, hidden = self.agent.get_action(
                self.gpu_loader.state_to_device(current_state))

        suppressed_termination = self.termination_helper.suppressed_termination(
            step, current_state, action) \
            if self.termination_helper and self.training else False

        if suppressed_termination:
            next_state, reward, done, _ = self.env.step(-1)
            done = True
        else:
            next_state, reward, done, _ = self.env.step(action)

        update_hidden(next_state, hidden)
        metrics['Rewards/ground_truth_reward'] = reward

        if trajectory is None:
            self.replay_buffer.append_step(action, reward, next_state, done,
                                           suppressed_termination=suppressed_termination)
        else:
            trajectory.append_step(action, reward, next_state, done,
                                   suppressed_termination=suppressed_termination)
        return metrics

    def generate(self, max_episode_length=100000, print_actions=False):
        trajectory = Trajectory()
        state = self.env.reset()
        trajectory.states.append(state)

        step = 0
        while not trajectory.done and len(trajectory) < max_episode_length:
            self.env_interaction_step(step, trajectory=trajectory)
            step += 1

        return trajectory

    def random_trajectories(self, steps, max_length=1000):
        print(f'Generating random trajectories for {steps} steps')
        self.start_new_trajectory()

        # generate random trajectories
        for step in range(steps):
            self.env_interaction_step(step % max_length, random_action=True)
            current_trajectory = self.replay_buffer.current_trajectory()
            if current_trajectory.suppressed_termination():
                self.start_new_trajectory(reset_env=False)
            elif current_trajectory.done or len(current_trajectory) > 1000:
                self.start_new_trajectory()

        trajectory_count = len(self.replay_buffer.trajectories)
        print(f'Finished generating {trajectory_count} random trajectories')
        return self.replay_buffer
