from networks.soft_q import SoftQNetwork
from networks.instrinsic_curiosity import CuriosityModule
from helpers.environment import ObservationSpace, ActionSpace
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from helpers.datasets import ReplayBuffer

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

import wandb


class SAC:
    def __init__(self,
                 reward_function=None,
                 run):
        self.alpha = run.config['alpha']
        self.n_observation_frames = run.config['n_observation_frames']
        self.discount_factor = run.config['discount_factor']
        self.actions = ActionSpace.actions()
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.actor = SoftQNetwork(n_observation_frames=self.n_observation_frames,
                                  alpha=self.alpha).to(self.device)
        self.target_q = SoftQNetwork(n_observation_frames=self.n_observation_frames,
                                     alpha=self.alpha).to(self.device)
        self.online_q = SoftQNetwork(n_observation_frames=self.n_observation_frames,
                                     alpha=self.alpha).to(self.device)
        self.curiosity_module = CuriosityModule(
            n_observation_frames=self.n_observation_frames).to(self.device)

    def train(self, env, run, profiler=None):
        self.curiosity_optimizer = th.optim.Adam(self.curiosity_module.parameters(),
                                                 lr=run.config['curiosity_lr'])
        self.policy_optimizer = th.optim.Adam(self.actor.parameters(),
                                              lr=run.config['policy_lr'])
        self.q_optimizer = th.optim.Adam(self.critic.parameters(),
                                         lr=run.config['q_lr'])
        self.run = run
        replay_buffer = ReplayBuffer(reward=True,
                                     n_observation_frames=self.n_observation_frames)

        # th.autograd.set_detect_anomaly(True)
        obs = env.reset()
        replay_buffer.current_trajectory().append_obs(obs)
        current_state = replay_buffer.current_state()

        for step in range(self.run.config['training_steps']):
            iter_count = step + 1
            if iter_count <= self.run.config['starting_steps']:
                action = self.random_action(
                    replay_buffer.current_trajectory().current_obs())
            else:
                action = self.get_action(current_state)

            replay_buffer.current_trajectory().actions.append(action)
            obs, _, done, _ = env.step(action)

            replay_buffer.current_trajectory().append_obs(obs)
            replay_buffer.current_trajectory().done = done

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
                q_loss, policy_loss, curiosity_loss, average_Q = self.train_one_batch(
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
            if (run.checkpoint_freqency and iter_count % run.checkpoint_freqency == 0
                    and iter_count < run.config['training_steps']):
                th.save(self.model.state_dict(), os.path.join('train', f'{run.name}.pth'))
                print(f'Checkpoint saved at step {iter_count}')

        print('Training complete')
        self.curiosity_optimizer = None
        self.policy_optimizer = None
        self.q_optimizer = None
        self.run = None

    def train_one_batch(self, batch):
        obs, actions, next_obs, done, rewards = batch
        states = ObservationSpace.obs_to_state(obs)
        next_states = ObservationSpace.obs_to_state(next_obs)
        all_states = [th.cat(state_component, dim=0).to(self.device) for state_component
                      in zip(states, next_states)]
        actions = actions.to(self.device)
        rewards = rewards.float().unsqueeze(1).to(self.device)

        # update critic
        all_Qs = self.critic.get_Q(all_states)
        Qs, next_Qs = th.chunk(all_Qs, 2, dim=0)
        next_Vs = self.critic.get_V(next_Qs)
        # use Qs only for taken actions
        Q_s_a = th.gather(Qs, dim=1, index=actions.unsqueeze(1))
        target_Qs = rewards + self.discount_factor * next_Vs
        q_loss = F.mse_loss(Q_s_a, target_Qs)

        # update actor
        entropies = self.actor.entropies(Qs)
        entropy_s_a = th.gather(entropies, dim=1, index=actions.unsqueeze(1))
        policy_loss = -th.mean(Q_s_a - self.alpha * entropy_s_a)

        # update curiosity module
        # loss for predicted action
        states, next_states = zip(*[th.chunk(state_component, 2, dim=0)
                                    for state_component in all_states])
        predicted_actions = self.curiosity_module.predict_action(states, next_states)
        action_loss = F.cross_entropy(predicted_actions, actions)

        # loss for predicted features
        current_features = self.curiosity_module.get_features(states)
        next_features = self.curiosity_module.get_features(next_states, single_frame=True)
        predicted_features = self.curiosity_module.predict_next_features(
            current_features, actions)
        feature_loss = F.mse_loss(predicted_features, next_features)

        curiosity_loss = action_loss + feature_loss

        total_loss = policy_loss + q_loss + curiosity_loss

        self.policy_optimizer.zero_grad(set_to_none=True)
        self.q_optimizer.zero_grad(set_to_none=True)
        self.curiosity_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.policy_optimizer.step()
        self.curiosity_optimizer.step()
        self.q_optimizer.step()

        return q_loss.detach(), policy_loss.detach(), \
            curiosity_loss.detach(), all_Qs.detach().mean().item()
