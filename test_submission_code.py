from agents.soft_q import SoftQAgent
from contexts.minerl.environment import ObservationWrapper, ActionShaping
from core.trajectory_generator import TrajectoryGenerator

import os
import time
import gym
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import torch as th


class EpisodeDone(Exception):
    pass


class Episode(gym.Env):
    """A class for a single episode."""

    def __init__(self, env):
        self.env = ActionShaping(env)
        self.env = ObservationWrapper(env)
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s, r, d, i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s, r, d, i


class MineRLAgent():
    """
    To compete in the competition, you are required to implement the two
    functions in this class:
        - load_agent: a function that loads e.g. network models
        - run_agent_on_episode: a function that plays one game of MineRL

    By default this agent behaves like a random agent: pick random action on
    each step.
    """

    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        with initialize(config_path='conf'):
            cfg = compose('config.yaml')

        cfg.device = "cuda:0" if th.cuda.is_available() else "cpu"
        cfg.wandb = False
        cfg.start_time = time.time()

        cfg.hydra_base_dir = os.getcwd()
        print(OmegaConf.to_yaml(cfg))
        environment = cfg.env.name
        os.environ['MINERL_ENVIRONMENT'] = environment

        self.model = SoftQAgent(cfg)
        for saved_agent_path in reversed(sorted(Path('train/').iterdir())):
            if saved_agent_path.suffix == '.pth' and environment in saved_agent_path.name:
                print(f'Loading {saved_agent_path.name} as agent')
                self.model.load_parameters(saved_agent_path)
                break

    def run_agent_on_episode(self, single_episode_env: Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs))
                ...

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        # An implementation of a random agent
        # YOUR CODE GOES HERE
        TrajectoryGenerator(single_episode_env, self.model, self.model.config).generate()
