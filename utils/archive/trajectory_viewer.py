from utils.environment import ObservationSpace, ActionSpace

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
