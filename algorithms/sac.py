from networks.soft_q import SoftQNetwork, TwinnedSoftQNetwork
from algorithms.loss_functions.sac import SACQLoss, SACPolicyLoss
from networks.intrinsic_curiosity import CuriosityModule
from helpers.environment import ObservationSpace, ActionSpace
from helpers.datasets import ReplayBuffer, MixedReplayBuffer
from helpers.gpu import states_to_device, disable_gradients

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

import wandb


class SoftActorCritic:
    def __init__(self, run,
                 actor=None,
                 expert_dataset=None):
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.run = run
        self.starting_steps = run.config['starting_steps']
        self.batch_size = run.config['batch_size']
        assert(self.starting_steps > self.batch_size)
        self.tau = run.config['tau']
        self.target_update_interval = 1
        self.updates_per_step = 1

        # Set up replay buffer
        if expert_dataset is None:
            self.replay_buffer = ReplayBuffer(
                reward=True, n_observation_frames=self.run.config['n_observation_frames'])
        else:
            self.replay_buffer = MixedReplayBuffer(
                expert_dataset=expert_dataset,
                batch_size=run.config['batch_size'],
                expert_sample_fraction=0.5,
                n_observation_frames=self.run.config['n_observation_frames'])

        # Set up networks - actor
        if actor is not None:
            self.actor = actor.to(self.device)
        else:
            self.actor = SoftQNetwork(
                n_observation_frames=run.config['n_observation_frames'],
                alpha=run.config['alpha']).to(self.device)

        # Set up networks - critic
        self.online_q = TwinnedSoftQNetwork(
            n_observation_frames=run.config['n_observation_frames'],
            alpha=run.config['alpha']).to(self.device)
        self.target_q = TwinnedSoftQNetwork(
            n_observation_frames=run.config['n_observation_frames'],
            alpha=run.config['alpha']).to(self.device)
        self.target_q.load_state_dict(self.online_q.state_dict())
        disable_gradients(self.target_q)

        self.curiosity_module = CuriosityModule(
            n_observation_frames=run.config['n_observation_frames']).to(self.device)

        # Loss functions
        self._q_loss = SACQLoss(self.online_q, self.target_q, run)
        self._policy_loss = SACPolicyLoss(self.actor, self.online_q, run)

        # Optimizers
        self.curiosity_optimizer = th.optim.Adam(self.curiosity_module.parameters(),
                                                 lr=run.config['curiosity_lr'])
        self.policy_optimizer = th.optim.Adam(self.actor.parameters(),
                                              lr=run.config['policy_lr'])
        self.q_optimizer = th.optim.Adam(self.online_q.parameters(),
                                         lr=run.config['q_lr'])

    def _soft_update_target(self):
        for target, online in zip(self.target_q.parameters(), self.online_q.parameters()):
            target.data.copy_(target.data * (1.0 - self.tau) + online.data * self.tau)

    def __call__(self, env, profiler=None):
        obs = env.reset()
        self.replay_buffer.current_trajectory().append_obs(obs)
        current_state = self.replay_buffer.current_state()

        for step in range(self.run.config['training_steps']):
            iter_count = step + 1
            if iter_count <= self.run.config['starting_steps']:
                action = self.actor.random_action(current_state)
            else:
                action = self.actor.get_action(current_state)

            self.replay_buffer.current_trajectory().actions.append(action)
            obs, _, done, _ = env.step(action)

            self.replay_buffer.current_trajectory().append_obs(obs)
            self.replay_buffer.current_trajectory().done = done

            # add elements of the transition tuple to the replay buffer individually
            # so we can use replay buffers current state to calculate the reward
            next_state = self.replay_buffer.current_state()

            if step < self.batch_size * 1.5:
                reward = .5
            else:
                reward = self.curiosity_module.reward(current_state, action,
                                                      next_state, done)

            self.replay_buffer.current_trajectory().rewards.append(reward)

            self.replay_buffer.increment_step()
            current_state = next_state

            doing_sac_updates = step >= self.starting_steps
            if step > self.batch_size:
                q_losses = []
                policy_losses = []
                curiosity_losses = []
                average_target_Qs = []
                average_online_Qs = []
                for i in range(self.updates_per_step):
                    q_loss, policy_loss, curiosity_loss, average_target_Q, \
                        average_online_Q = self.train_one_batch(
                            self.replay_buffer.sample(batch_size=self.batch_size),
                            do_sac_updates=doing_sac_updates)
                    if self.run.wandb:
                        q_losses = q_losses.append(q_loss)
                        policy_losses = policy_losses.append(policy_loss)
                        curiosity_losses = curiosity_losses.append(curiosity_loss)
                        average_target_Qs = average_target_Qs.append(average_target_Q)
                        average_online_Q = average_online_Qs.append(average_online_Q)
                if self.run.wandb:
                    wandb.log({'reward': reward,
                               'policy_loss': mean(policy_losses),
                               'q_loss': mean(q_losses),
                               'curiosity_loss': mean(curiosity_losses),
                               'average_target_Qs': mean(average_target_Qs),
                               'average_online_Qs': mean(average_online_Qs),
                               'average_its_per_s': self.run.iteration_rate()})

            if doing_sac_updates and step % self.target_update_interval:
                self._soft_update_target()

            if done:
                print(f'Trajectory completed at step {iter_count}')
                self.replay_buffer.new_trajectory()
                obs = env.reset()
                self.replay_buffer.current_trajectory().append_obs(obs)
                current_state = self.replay_buffer.current_state()

            self.run.step()
            self.run.print_update()

            if profiler:
                profiler.step()

            if self.run.checkpoint_freqency \
                    and iter_count % self.run.checkpoint_freqency == 0 \
                    and iter_count < run.config['training_steps']:
                self.save(os.path.join('train', f'{self.run.name}.pth'))
                print(f'Checkpoint saved at step {iter_count}')

        print('Training complete')

    def train_one_batch(self, batch, do_sac_updates=True):
        # load batch onto gpu
        obs, actions, next_obs, _done, rewards = batch
        states = ObservationSpace.obs_to_state(obs)
        next_states = ObservationSpace.obs_to_state(next_obs)
        states, next_states = states_to_device((states, next_states), self.device)
        actions = actions.to(self.device)
        if do_sac_updates:
            rewards = rewards.float().unsqueeze(1).to(self.device)
        batch = states, actions, next_states, _done, rewards

        if do_sac_updates:
            # update q
            q_loss, average_target_Qs = self._q_loss(*batch)
            self.q_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            self.q_optimizer.step()
            q_loss = q_loss.detach()

            # update policy
            policy_loss, average_online_Qs = self._policy_loss(*batch)
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            self.policy_optimizer.step()
            policy_loss = policy_loss.detach()
        else:
            q_loss = 0
            policy_loss = 0
            average_target_Q = 0
            average_online_Q = 0

        # update curiosity
        curiosity_loss = self.curiosity_module.loss(*batch)
        self.curiosity_optimizer.zero_grad(set_to_none=True)
        curiosity_loss.backward()
        self.curiosity_optimizer.step()
        curiosity_loss = curiosity_loss.detach()

        return q_loss, policy_loss, curiosity_loss, average_target_Q, average_online_Q

    def save(self, save_path):
        Path(save_path).mkdir(exist_ok=True)
        self.actor.save(os.path.join(save_path, 'actor.pth'))
        self.target_q.save(os.path.join(save_path, 'critc.pth'))
