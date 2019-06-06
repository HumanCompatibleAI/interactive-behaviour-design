import os
from abc import abstractmethod
from enum import Enum
from threading import Thread

import easy_tf_log
import numpy as np

from global_constants import DEFAULT_BC_COEF
from rollouts import RolloutsByHash

"""
The base Policy class is designed in a bit of a strange way.
Why do we have init_logger and set_training_env? Why not just initialise these at the start?

The reason is that the rollout workers need to be able to initialise policies without also
involving all the extra crap. Specifically, rollout workers shouldn't do any logging, and we
should try and pickle the environment when we pickle the 'make policy function'.

So the Policy constructor itself is fairly minimal, only setting up the step model.
Other stuff like runners and loggers are started later on when needed.
"""


class Policy:
    def __init__(self, name, env_id, obs_space, ac_space, n_envs, seed=0):
        self.name = name
        # Important that this starts out as NO_TRAINING so that we don't make the policy step counter
        # non-None when we're only pre-training, so that the preference rate limiter doesn't kick in
        self.train_mode = PolicyTrainMode.NO_TRAINING
        self.bc_coef = DEFAULT_BC_COEF
        self.n_updates = 0
        self.log_interval = 20
        self.logger = None  # type: easy_tf_log.Logger
        self.training_enabled = False
        self.train_thread = None
        self.demonstration_rollouts = None
        self.n_total_steps = None

    def init_logger(self, log_dir):
        if self.logger is None:
            self.logger = easy_tf_log.Logger()
            self.logger.set_log_dir(os.path.join(log_dir, 'policy_{}'.format(self.name)))

    def start_training(self):
        self.training_enabled = True
        self.train_thread = Thread(target=self.train_loop)
        self.train_thread.start()

    def stop_training(self):
        if self.train_thread:
            self.training_enabled = False
            self.train_thread.join()

    def train_loop(self):
        while self.training_enabled:
            self.train()

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self, obs, **step_kwargs):
        raise NotImplementedError()

    @abstractmethod
    def load_checkpoint(self, path):
        raise NotImplementedError()

    @abstractmethod
    def save_checkpoint(self, path):
        raise NotImplementedError()

    @abstractmethod
    def set_training_env(self, env, log_dir):
        raise NotImplementedError()

    @abstractmethod
    def set_test_env(self, env, log_dir):
        raise NotImplementedError()

    @abstractmethod
    def use_demonstrations(self, demonstrations: RolloutsByHash):
        raise NotImplementedError()


class PolicyTrainMode(Enum):
    R_ONLY = 1
    R_PLUS_BC = 2
    BC_ONLY = 3
    SQIL_ONLY = 4
    R_PLUS_SQIL = 5
    NO_TRAINING = 6


class EpisodeRewardLogger():
    def __init__(self, log_dir, n_steps, n_envs):
        self.logger = easy_tf_log.Logger(os.path.join(log_dir, 'processed_reward_logger'))
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.reward = 0

    def log(self, mb_rewards, mb_dones):
        mb_rewards = np.array(mb_rewards)
        mb_dones = np.array(mb_dones)
        assert mb_rewards.shape == (self.n_steps, self.n_envs)
        assert mb_dones.shape == (self.n_steps, self.n_envs)
        mb_rewards_env0 = mb_rewards[:, 0]
        mb_dones_env0 = mb_dones[:, 0]
        for n in range(len(mb_rewards)):
            self.reward += mb_rewards_env0[n]
            if mb_dones_env0[n]:
                self.logger.logkv('processed_reward', self.reward)
                self.reward = 0
