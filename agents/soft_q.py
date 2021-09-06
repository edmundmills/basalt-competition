import torch as th
import torch.nn.functional as F
import numpy as np
from models.soft_q_network import SoftQNetwork
from environment.observation_space import dataset_obs_batch_to_obs
from environment.action_space import dataset_action_batch_to_actions


class SoftQAgent:
    def __init__(self, discount_factor=.99, alpha=1):
        self.actions = ActionSpace.actions()
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.model = SoftQNetwork(actions=self.actions, alpha=self.alpha).to(self.device)

    def load_parameters(self, model_file_path):
        self.model.load_state_dict(
            th.load(model_file_path, map_location=self.device), strict=False)

    def save(self, path):
        th.save(agent.model.state_dict(), path)

    def get_action(self, observations, supress_snowball=False):
        with th.no_grad():
            Q = self.model.get_Q(observations)
            probabilities = self.model.action_probabilities(Q).cpu().numpy().squeeze()

        action = np.random.choice(self.actions, p=probabilities)
        while action == 11 and supress_snowball:
            action = np.random.choice(self.actions, p=probabilities)
        return action

    def train(self, env, expert_data_path, run, training_steps):
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=run.lr)

        replay_buffer = MixedReplayBuffer(capacity=10000000, batch_size=64,
                                          expert_data_path=expert_data_path,
                                          expert_sample_fraction=0.5)

        obs = env.reset()
        initial_obs = obs_tensor_from_single_obs(obs)
        trajectory_step = 0

        for _, step in tqdm(enumerate(range(training_steps))):
            supress_snowball = trajectory_step < 100
            action = agent.get_action(initial_obs.to(device),
                                      supress_snowball=supress_snowball)
            if action == 11:
                print(f'Threw Snowball at step {step}')
            next_obs, _, done, _ = env.step(action)
            trajectory_step += 1
            next_obs = obs_tensor_from_single_obs(next_obs)
            replay_buffer.push(initial_obs.detach().clone(),
                               action.copy(),
                               next_obs.detach().clone(),
                               done)

            if len(replay_buffer) >= replay_buffer.replay_batch_size:
                loss = agent.train_one_batch(replay_buffer.sample_expert(),
                                             replay_buffer.sample_replay())
                run.append_loss(loss.detach().item())

            if done:
                print(f'Trajectory completed at step {step}')
                obs = env.reset()
                initial_obs = obs_tensor_from_single_obs(obs)
                trajectory_step = 0
            else:
                initial_obs = next_obs

        self.save()


class SoftQNetwork(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.actions = ActionSpace.actions()
        self.alpha = alpha
        self.cnn = mobilenet_v3_large(pretrained=True, progress=True).features

        self.linear = nn.Sequential(
            nn.Linear(self.vision_module.features_output_dim(), 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, len(self.actions))
        )

    def forward(self, observations: th.Tensor, grad=False) -> th.Tensor:
        features = th.flatten(
            self.vision_module.extract_features(observations, grad=grad), 1)
        return self.linear(features)

    def get_Q(self, observations, grad=False):
        return self.forward(observations, grad=grad).clamp(-32, 32)

    def get_V(self, Qs):
        v = self.alpha * \
            th.log(th.sum(th.exp(Qs / self.alpha), dim=1, keepdim=True))
        return v

    def action_probabilities(self, Qs):
        Vs = self.get_V(Qs).repeat(1, len(self.actions))
        probabilities = th.exp((Qs - Vs)/self.alpha)
        probabilities /= th.sum(probabilities)
        return probabilities


class SqilAgent(SoftQAgent):
    def train_one_batch(self, expert_batch, replay_batch):
        expert_states, expert_actions, expert_next_states, expert_done = expert_batch
        replay_states, replay_actions, replay_next_states, replay_done = replay_batch
        replay_states, replay_actions, replay_next_states, replay_done = (
            replay_states.to(self.device), replay_actions.to(self.device),
            replay_next_states.to(self.device), replay_done.to(self.device))
        expert_actions = dataset_action_batch_to_actions(expert_actions)
        expert_states = dataset_obs_batch_to_obs(expert_states).to(self.device)
        expert_next_states = dataset_obs_batch_to_obs(expert_next_states).to(self.device)

        # remove expert no-op actions
        mask = expert_actions != -1
        expert_states = expert_states[mask]
        expert_actions = expert_actions[mask]
        expert_next_states = expert_next_states[mask]
        expert_done = expert_done[mask]
        masked_expert_batch_size = len(expert_actions)
        replay_batch_size = len(replay_actions)

        expert_actions = th.from_numpy(expert_actions).unsqueeze(1).to(self.device)

        expert_rewards = th.ones(masked_expert_batch_size, 1)
        replay_rewards = th.ones(replay_batch_size, 1)
        batch_rewards = th.cat([expert_rewards, replay_rewards], dim=0).to(self.device)
        expert_done = expert_done.long().unsqueeze(1).to(self.device)
        batch_done = th.cat([expert_done, replay_done], dim=0)
        batch_actions = th.cat([expert_actions, replay_actions], dim=0)
        batch_states = th.cat([expert_states, replay_states,
                              expert_next_states, replay_next_states], dim=0)

        batch_Qs = self.model.get_Q(batch_states, grad=True)
        current_Qs, next_Qs = th.chunk(batch_Qs, 2, dim=0)

        Q_s_a = th.gather(current_Qs, 1, batch_actions)
        V_next = self.model.get_V(next_Qs)
        y = batch_rewards + self.gamma * V_next * (1 - batch_done)

        objective = F.mse_loss(Q_s_a, y)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        return objective.detach()


class IQLearnAgent(SoftQAgent):
    def train_one_batch(self, expert_batch, replay_batch):
        def distance_function(x):
            return x - 1/4 * x**2

        expert_states, expert_actions, expert_next_states, _expert_done = expert_batch
        replay_states, replay_actions, replay_next_states, _replay_done = replay_batch
        replay_states, replay_actions, replay_next_states = (
            replay_states.to(self.device), replay_actions.to(self.device),
            replay_next_states.to(self.device))
        expert_actions = dataset_action_batch_to_actions(expert_actions)
        expert_states = dataset_obs_batch_to_obs(expert_states).to(self.device)
        expert_next_states = dataset_obs_batch_to_obs(expert_next_states).to(self.device)

        # remove expert no-op actions
        mask = expert_actions != -1
        expert_states = expert_states[mask]
        expert_actions = expert_actions[mask]
        expert_next_states = expert_next_states[mask]
        masked_expert_batch_size = len(expert_actions)
        replay_batch_size = len(replay_actions)

        all_states = th.cat([expert_states, expert_next_states,
                            replay_states, replay_next_states], dim=0)
        all_Qs = self.model.get_Q(all_states, grad=True)
        expert_Qs, expert_Qs_next, replay_Qs, replay_Qs_next = th.split(
            all_Qs, [masked_expert_batch_size, masked_expert_batch_size,
                     replay_batch_size, replay_batch_size], dim=0)

        Q_expert_s_a = expert_Qs[th.arange(len(expert_actions)),
                                 expert_actions].unsqueeze(1)
        V_next_expert = self.model.get_V(expert_Qs_next)
        V_replay = self.model.get_V(replay_Qs)
        V_replay_next = self.model.get_V(replay_Qs_next)

        objective = -(th.mean(distance_function(Q_expert_s_a - self.gamma *
                      V_next_expert)) - th.mean(V_replay - self.gamma * V_replay_next))

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        return objective.detach()
