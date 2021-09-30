from helpers.environment import ObservationSpace, ActionSpace

import math
import os
import shutil
from pathlib import Path
from collections import OrderedDict, deque

import numpy as np
import torch as th

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import cv2


class Trajectory:
    def __init__(self, n_observation_frames):
        self.n_observation_frames = n_observation_frames
        self.framestack = deque(maxlen=self.n_observation_frames)
        self.states = []
        self.actions = []
        self.rewards = []
        self.done = False
        self.additional_data = OrderedDict()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        is_last_step = idx + 1 == len(self)
        done = is_last_step and self.done
        if done:
            next_state = self.states[idx]
        elif is_last_step:
            next_state = None
        else:
            next_state = self.states[idx + 1]
        reward = self.rewards[idx] if len(self.rewards) > 0 else 0
        return self.states[idx], self.actions[idx], next_state, done, reward

    def append_obs(self, obs):
        pov = ObservationSpace.obs_to_pov(obs)
        # initialize framestack
        while len(self.framestack) < self.n_observation_frames:
            self.framestack.append(pov)
        self.framestack.append(pov)
        pov = th.cat(list(self.framestack), dim=0)
        items = ObservationSpace.obs_to_items(obs)
        state = pov, items
        self.states.append(state)

    def current_state(self, **kwargs):
        current_idx = len(self) - 1
        state = self.states[current_idx]
        return state

    def get_pov(self, idx):
        pov, _items = self.states[idx]
        single_frame = pov[-3:, :, :]
        return single_frame

    def save(self, path):
        path.mkdir(exist_ok=True)
        np.save(file=path / 'actions.npy', arr=np.array(self.actions))
        np.save(file=path / 'obs.npy', arr=np.array(self.obs))
        if len(self.additional_data) > 0:
            np.save(file=path / 'additional_data.npy',
                    arr=np.array(self.additional_data))

        steps_path = path / 'steps'
        shutil.rmtree(steps_path, ignore_errors=True)
        steps_path.mkdir()
        for step in range(len(self)):
            obs, action, next_obs, done = self[step]
            step_name = f'step{str(step).zfill(5)}.npy'
            step_dict = {'step': step, 'obs': obs, 'action': action, 'done': done}
            np.save(file=steps_path / step_name, arr=step_dict)

    def save_as_video(self, save_dir_path, filename):
        save_dir_path = Path(save_dir_path)
        save_dir_path.mkdir(exist_ok=True)
        images, frame_rate = self.as_video_frames()
        video_path = save_dir_path / f'{filename}.mp4'
        frame_size = (64, 64)
        out = cv2.VideoWriter(str(video_path),
                              cv2.VideoWriter_fourcc(*'FMP4'),
                              frame_rate, frame_size)
        for img in images:
            out.write(img)
        out.release()
        return video_path

    def as_video_frames(self):
        total_steps = len(self)
        frame_skip = 2
        frames = min(int(round(total_steps / (frame_skip + 1))), total_steps)
        step_rate = 20  # steps / second
        frame_rate = int(round(step_rate / (frame_skip + 1)))
        duration = frames / frame_rate
        step_indices = [frame * (frame_skip + 1) for frame in range(frames)]
        images = [(self.get_pov(idx).numpy()).astype(
            np.uint8).transpose(1, 2, 0)[..., ::-1]
            for idx in step_indices]
        return images, frame_rate

    def view(self):
        viewer = TrajectoryViewer(self).view()


class TrajectoryGenerator:
    def __init__(self, env, replay_buffer=None):
        self.env = env
        self.replay_buffer = replay_buffer

    def generate(self, model, max_episode_length=100000):
        trajectory = Trajectory(n_observation_frames=self.model.n_observation_frames)
        obs = self.env.reset()

        while not trajectory.done and len(trajectory) < max_episode_length:
            trajectory.append_obs(obs)
            state = trajectory.current_state()
            action = model.get_action(state)
            trajectory.actions.append(action)
            obs, _, done, _ = self.env.step(action)
            trajectory.done = done
        return trajectory

    def new_trajectory(env, replay_buffer, reset_env=True, current_obs=None):
        if len(replay_buffer.current_trajectory()) > 0:
            replay_buffer.new_trajectory()
        if reset_env:
            obs = env.reset()
        else:
            obs = current_obs
        replay_buffer.current_trajectory().append_obs(obs)
        current_state = replay_buffer.current_state()
        return current_state

    def start_new_trajectory(self, **kwargs):
        current_state = TrajectoryGenerator.new_trajectory(self.env, self.replay_buffer,
                                                           **kwargs)
        return current_state

    def random_trajectories(self, steps):
        print(f'Generating random trajectories for {steps} steps')

        current_state = self.start_new_trajectory()

        # generate random trajectories
        for step in range(steps):
            action = ActionSpace.random_action()
            self.replay_buffer.current_trajectory().actions.append(action)

            suppressed_snowball = ActionSpace.threw_snowball(current_state, action)
            if suppressed_snowball:
                obs, _, done, _ = self.env.step(-1)
                reward = -1
            else:
                obs, _, done, _ = self.env.step(action)
                reward = 0

            self.replay_buffer.current_trajectory().append_obs(obs)
            self.replay_buffer.current_trajectory().done = done
            next_state = self.replay_buffer.current_state()

            self.replay_buffer.current_trajectory().rewards.append(reward)

            self.replay_buffer.increment_step()
            current_state = next_state

            if done or (step % 1000 == 0 and step != steps):
                print(f'Random trajectory completed at step {step}')
                current_state = self.start_new_trajectory()
            elif suppressed_snowball:
                current_state = self.start_new_trajectory(reset_env=False,
                                                          current_obs=obs)

        trajectory_count = len(self.replay_buffer.trajectories)
        print(f'Finished generating {trajectory_count} random trajectories')
        return self.replay_buffer


class TrajectoryViewer:
    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.current_step = 0
        self.manual_control = False

    def view(self):
        def play():
            if len(self.trajectory) > 1:
                slider.set_val(self.current_step)
                slider.on_changed(update_slider)
                fig.canvas.mpl_connect('button_press_event', on_click)
                ani = animation.FuncAnimation(
                    fig, update_plot, animation_data_gen, interval=2, blit=True)
            fig.canvas.mpl_connect('key_press_event', on_keypress)
            plt.show()

        def animation_data_gen():
            for step, _ in enumerate(self.trajectory):
                yield step

        def update_plot(step):
            if self.manual_control:
                return animated_elements
            step = slider.val + 1
            if step >= len(self.trajectory):
                step = len(self.trajectory) - 1
                slider.set_val(step)
                return animated_elements
            slider.set_val(step)
            self.manual_control = False
            return animated_elements

        def update_slider(step):
            step = math.floor(step)
            self.current_step = step
            self.manual_control = True
            render_frame(step)

        def render_frame(step):
            frame = self.trajectory.obs[step]["pov"]
            action = self.trajectory.actions[step]
            if not isinstance(action, (int, np.int64)):
                action = ActionSpace.dataset_action_batch_to_actions(action)[0]
            action_name = ActionSpace.action_name(action)
            txt_action.set_text(f'Action: {action_name}')
            img.set_array(frame)
            if first_plot_marker:
                first_plot_marker.set_xdata(step)
            if second_plot_marker:
                second_plot_marker.set_xdata(step)
            fig.canvas.flush_events()
            fig.canvas.draw_idle()

        def on_click(event):
            (sxm, sym), (sxM, syM) = slider.label.clipbox.get_points()
            if sxm < event.x < sxM and sym < event.y < syM:
                return
            self.manual_control = False if self.manual_control else True

        def on_keypress(event):
            if event.key == 'enter':
                plt.close('all')
                print('Viewer Closed')
            if len(self.trajectory) < 2:
                return
            if event.key == ' ':
                self.manual_control = False if self.manual_control else True
            else:
                self.manual_control = True
                if event.key == 'right':
                    if self.current_step < len(self.trajectory) - 1:
                        self.current_step += 1
                elif event.key == 'left':
                    if self.current_step > 0:
                        self.current_step -= 1
                slider.set_val(self.current_step)

        fig = plt.figure(figsize=(9, 8))
        ax_pov = plt.subplot2grid((9, 8), (0, 0), colspan=5, rowspan=5)
        ax_pov.get_xaxis().set_visible(False)
        ax_pov.get_yaxis().set_visible(False)
        img = ax_pov.imshow(self.trajectory.obs[self.current_step]["pov"], animated=True)
        ax_text = plt.subplot2grid((9, 8), (6, 0), colspan=8, rowspan=2)
        ax_text.get_xaxis().set_visible(False)
        ax_text.get_yaxis().set_visible(False)
        txt_action = ax_text.text(.1, 1.35, 'Action:')
        animated_elements = [img, txt_action]
        if len(list(self.trajectory.additional_data.keys())) > 0:
            ax_first_plot = plt.subplot2grid((9, 8), (0, 6), colspan=4, rowspan=2)
            ax_first_plot.set(ylabel=list(self.trajectory.additional_data.keys())[0],
                              xlim=(0, len(self.trajectory)))
            ax_first_plot.plot(range(len(self.trajectory)),
                               list(self.trajectory.additional_data.values())[0], 'b-')
            first_plot_marker = ax_first_plot.axvline(x=-1, color='r')
            animated_elements.append(first_plot_marker)
        else:
            first_plot_marker = None
        if len(list(self.trajectory.additional_data.keys())) > 1:
            ax_second_plot = plt.subplot2grid((9, 8), (3, 6), colspan=4, rowspan=2)
            ax_second_plot.set(ylabel=list(self.trajectory.additional_data.keys())[1],
                               xlim=(0, len(self.trajectory)))
            ax_second_plot.plot(range(len(self.trajectory)),
                                list(self.trajectory.additional_data.values())[1], 'b-')
            second_plot_marker = ax_second_plot.axvline(x=-1, color='r')
            animated_elements.append(second_plot_marker)
        else:
            second_plot_marker = None

        if len(self.trajectory) > 0:
            ax_steps = plt.axes([0.2, 0.15, 0.65, 0.03])
            slider = Slider(ax_steps, 'Step', 0, len(self.trajectory) - 1, valinit=0)

        play()
