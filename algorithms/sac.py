from networks.soft_q import SoftQNetwork
from algorithms.loss_functions.sac import SACQLoss, SACPolicyLoss
from networks.instrinsic_curiosity import CuriosityModule
from helpers.environment import ObservationSpace, ActionSpace
from helpers.datasets import ReplayBuffer, MixedReplayBuffer

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

import wandb


class SAC:
    def __init__(self,
                 actor=None,
                 run):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        if actor is not None:
            self.actor = actor.to(self.device)
        else:
            self.actor = SoftQNetwork(
                n_observation_frames=run.config['n_observation_frames'],
                alpha=run.config['alpha']).to(self.device)
        self.target_q = SoftQNetwork(
            n_observation_frames=run.config['n_observation_frames'],
            alpha=run.config['alpha']).to(self.device)
        self.online_q = SoftQNetwork(
            n_observation_frames=run.config['n_observation_frames'],
            alpha=run.config['alpha']).to(self.device)
        self.curiosity_module = CuriosityModule(
            n_observation_frames=run.config['n_observation_frames']).to(self.device)
        self._q_loss = SACQLoss(self.online_q, run)
        self._policy_loss = SACPolicyLoss(self.actor, self.target_q, run)
        self.curiosity_optimizer = th.optim.Adam(self.curiosity_module.parameters(),
                                                 lr=run.config['curiosity_lr'])
        self.policy_optimizer = th.optim.Adam(self.actor.parameters(),
                                              lr=run.config['policy_lr'])
        self.q_optimizer = th.optim.Adam(self.online_q.parameters(),
                                         lr=run.config['q_lr'])
        self.run = run

    def __call__(self, env, profiler=None):
        replay_buffer = ReplayBuffer(
            reward=True, n_observation_frames=self.run.config['n_observation_frames'])

        obs = env.reset()
        replay_buffer.current_trajectory().append_obs(obs)
        current_state = replay_buffer.current_state()

        for step in range(self.run.config['training_steps']):
            iter_count = step + 1
            if iter_count <= self.run.config['starting_steps']:
                action = self.actor.random_action(current_state)
            else:
                action = self.actor.get_action(current_state)

            replay_buffer.current_trajectory().actions.append(action)
            obs, _, done, _ = env.step(action)

            replay_buffer.current_trajectory().append_obs(obs)
            replay_buffer.current_trajectory().done = done

            # add things to the replay buffer individually,
            # so we can use this, which is needed to calculate the reward
            next_state = replay_buffer.current_state()

            if step < self.run.config['starting_steps']:
                reward = 0
            else:
                reward = self.curiosity_module.reward(
                    current_state, action, next_state, done)
            if run.wandb:
                wandb.log({'reward': reward})

            replay_buffer.current_trajectory().rewards.append(reward)

            replay_buffer.increment_step()
            current_state = next_state

            batch_size = run.config['batch_size']
            if len(replay_buffer) >= batch_size:
                q_loss, policy_loss, curiosity_loss = self.train_one_batch(
                    replay_buffer.sample(batch_size=batch_size))
                if run.wandb:
                    wandb.log({'policy_loss': policy_loss,
                               'q_loss': q_loss,
                               'curiosity_loss': curiosity_loss,
                               'average_Q': average_Q})

            if done:
                print(f'Trajectory completed at step {iter_count}')
                replay_buffer.new_trajectory()
                obs = env.reset()
                replay_buffer.current_trajectory().append_obs(obs)
                current_state = replay_buffer.current_state()

            if profiler:
                profiler.step()
            if run.checkpoint_freqency and iter_count % run.checkpoint_freqency == 0 \
                    and iter_count < run.config['training_steps']:
                self.save(os.path.join('train', f'{run.name}.pth'))
                print(f'Checkpoint saved at step {iter_count}')

        print('Training complete')

    def train_one_batch(self, batch):
        obs, actions, next_obs, done, rewards = batch
        states = ObservationSpace.obs_to_state(obs)
        next_states = ObservationSpace.obs_to_state(next_obs)
        all_states = [th.cat(state_component, dim=0).to(self.device) for state_component
                      in zip(states, next_states)]
        actions = actions.to(self.device)
        rewards = rewards.float().unsqueeze(1).to(self.device)

        q_loss = self._q_loss(all_states, actions)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        policy_loss = self._policy_loss(states)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        curiosity_loss = self.curiosity_module.loss(states, actions, next_states, done)
        self.curiosity_optimizer.zero_grad(set_to_none=True)
        curiosity_loss.backward()
        self.curiosity_optimizer.step()

        return q_loss.detach(), policy_loss.detach(), curiosity_loss.detach()

    def save(self, save_path):
        Path(save_path).mkdir(exist_ok=True)
        self.actor.save(os.path.join(save_path, 'actor.pth'))
        self.target_q.save(os.path.join(save_path, 'critc.pth'))
