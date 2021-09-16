from helpers.environment import ObservationSpace, ActionSpace


class Algorithm:
    def generate_random_trajectories(self, replay_buffer, env, steps):
        obs = env.reset()
        replay_buffer.current_trajectory().append_obs(obs)
        current_state = replay_buffer.current_state()

        # generate random trajectories
        for step in range(steps):
            iter_count = step + 1
            action = ActionSpace.random_action()
            replay_buffer.current_trajectory().actions.append(action)

            if ActionSpace.threw_snowball(current_state, action):
                print('Snowball suppressed')
                obs, _, done, _ = env.step(-1)
            else:
                obs, _, done, _ = env.step(action)

            replay_buffer.current_trajectory().append_obs(obs)
            replay_buffer.current_trajectory().done = done
            next_state = replay_buffer.current_state()

            reward = 0
            replay_buffer.current_trajectory().rewards.append(reward)

            replay_buffer.increment_step()
            current_state = next_state

            if done or (iter_count % 1000 == 0 and iter_count != steps):
                print(f'Starting trajectory completed at step {iter_count}')
                replay_buffer.new_trajectory()
                obs = env.reset()
                replay_buffer.current_trajectory().append_obs(obs)
                current_state = replay_buffer.current_state()

            self.run.step()
            self.run.print_update()
