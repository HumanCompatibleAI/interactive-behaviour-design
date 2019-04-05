import os
from abc import abstractmethod
from enum import Enum
from threading import Thread

import easy_tf_log

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
        self.train_mode = PolicyTrainMode.R_ONLY
        self.bc_coef = DEFAULT_BC_COEF
        self.n_updates = 0
        self.log_interval = 20
        self.logger = None  # type: easy_tf_log.Logger
        self.training_enabled = False
        self.train_thread = None
        self.demonstration_rollouts = None

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
            self.n_updates += 1
            if self.n_updates % self.log_interval == 0:
                self.logger.logkv('policy_{}/n_updates'.format(self.name), self.n_updates)

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
    def set_training_env(self, env):
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
