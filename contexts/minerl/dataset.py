from core.trajectories import Trajectory
from contexts.minerl.environment import MineRLContext, start_env

import os
from pathlib import Path

import minerl
import numpy as np


class MineRLDatasetBuilder:
    def __init__(self, config):
        self.data_root = Path(os.getenv('MINERL_DATA_ROOT'))
        self.environment = config.env.name
        self.context = MineRLContext(config)
        self.environment_path = self.data_root / self.environment
        self.camera_margin = config.context.camera_margin

    def _dataset_obs_to_state(self, dataset_obs):
        state = self.obs_processor.observation(dataset_obs)
        return state

    def _dataset_action_to_action(self, dataset_action):
        camera_actions = dataset_actions["camera"].reshape((-1, 2))
        attack_actions = dataset_actions["attack"].reshape(-1)
        forward_actions = dataset_actions["forward"].reshape(-1)
        back_actions = dataset_actions["back"].reshape(-1)
        left_actions = dataset_actions["left"].reshape(-1)
        right_actions = dataset_actions["right"].reshape(-1)
        jump_actions = dataset_actions["jump"].reshape(-1)
        if not self.context.items_available:
            equip_actions = dataset_actions["equip"]
            use_actions = dataset_actions["use"].reshape(-1)

        batch_size = len(attack_actions)
        actions = np.zeros((batch_size,), dtype=np.int32)

        for i in range(batch_size):
            if not self.context.items_available and use_actions[i] == 1:
                actions[i] = self.context.action_name_list.index('Use')
            elif self.items_available and equip_actions in self.items:
                actions[i] = len(self.actions) - 1 + self.items.index(equip_actions)
            elif camera_actions[i][0] < -self.camera_margin:
                actions[i] = self.context.action_name_list.index('Look Up')
            elif camera_actions[i][0] > self.camera_margin:
                actions[i] = self.context.action_name_list.index('Look Down')
            elif camera_actions[i][1] > self.camera_margin:
                actions[i] = self.context.action_name_list.index('Look Right')
            elif camera_actions[i][1] < -self.camera_margin:
                actions[i] = self.context.action_name_list.index('Look Left')
            elif forward_actions[i] == 1 and jump_actions[i] == 1:
                actions[i] = self.context.action_name_list.index('Forward Jump')
            elif forward_actions[i] == 1 and jump_actions[i] != 1:
                actions[i] = self.context.action_name_list.index('Forward')
            elif attack_actions[i] == 1:
                actions[i] = self.context.action_name_list.index('Attack')
            elif jump_actions[i] == 1:
                actions[i] = self.context.action_name_list.index('Jump')
            elif back_actions[i] == 1:
                actions[i] = self.context.action_name_list.index('Back')
            elif left_actions[i] == 1:
                actions[i] = self.context.action_name_list.index('Left')
            elif right_actions[i] == 1:
                actions[i] = self.context.action_name_list.index('Right')
            else:
                actions[i] = -1
        return actions

    def load_data(self):
        data = minerl.data.make(self.environment)
        trajectories = []
        step_lookup = []

        trajectory_paths = list(self.environment_path.iterdir())
        if self.environment == 'MineRLBasaltCreateVillageAnimalPen-v0':
            animal_pen_plains_path = \
                self.environment_path / 'MineRLBasaltCreateAnimalPenPlains-v0'
            trajectory_paths.extend(list(animal_pen_plains_path.iterdir()))
        trajectory_idx = 0
        for trajectory_path in trajectory_paths:
            if not trajectory_path.is_dir():
                continue
            if trajectory_path.name in [
                    'v3_villainous_black_eyed_peas_loch_ness_monster-2_95372-97535',
                    'MineRLBasaltCreateAnimalPenPlains-v0']:
                continue

            trajectory = Trajectory()
            step_idx = 0
            print(trajectory_path)
            for obs, action, reward, _next_obs, done \
                    in data.load_data(str(trajectory_path)):
                trajectory.done = done
                action = self._dataset_action_to_action(action)[0]
                if action == -1:
                    continue
                state = self._dataset_obs_to_state(obs)
                trajectory.states.append(state)
                trajectory.actions.append(action)
                trajectory.rewards.append(reward)
                step_lookup.append((trajectory_idx, step_idx))
                step_idx += 1
            print(f'Loaded data from {trajectory_path.name} ({step_idx} steps)')
            trajectories.append(trajectory)
            trajectory_idx += 1
            if self.debug_dataset and trajectory_idx >= 2:
                break
            if self.environment in ['MineRLTreechop-v0', 'MineRLNavigateDense-v0',
                                    'MineRLNavigateExtremeDense-v0'] \
                    and trajectory_idx >= 80:
                break
        return trajectories, step_lookup
