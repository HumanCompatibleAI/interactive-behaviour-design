import multiprocessing
import os
from abc import abstractmethod
from collections import namedtuple
from enum import Enum
from threading import Thread

import easy_tf_log
import numpy as np
from gym.utils.atomic_write import atomic_write
from matplotlib.pyplot import figure, clf, plot, savefig

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
        self.n_serial_steps = 0
        self.n_envs = n_envs
        self.log_dir = None

    def init_logger(self, log_dir):
        if self.logger is None:
            self.logger = easy_tf_log.Logger()
            self.logger.set_log_dir(os.path.join(log_dir, 'policy_{}'.format(self.name)))
            self.log_dir = log_dir

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
            if self.n_serial_steps > 100:
                # Only write this once we've really started training;
                # throttler uses this to decide when to start throttling
                self.write_n_total_steps()

    def n_total_steps(self):
        return self.n_serial_steps * self.n_envs

    def write_n_total_steps(self):
        fname = os.path.join(self.log_dir, f'policy_{self.name}', 'n_total_steps.txt')
        # Needs to be atomic because is read by throttler, possibly in a different thread
        with atomic_write(fname) as f:
            f.write(str(self.n_total_steps()) + '\n')

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


Rewards = namedtuple('Rewards', 'env_n env_rewards predicted_rewards')


class EpisodeRewardLogger():
    def __init__(self, log_dir, n_steps, n_envs):
        self.logger = easy_tf_log.Logger(os.path.join(log_dir, 'processed_reward_logger'))
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.env_rewards_by_env_n = [[] for _ in range(n_envs)]
        self.predicted_rewards_by_env_n = [[] for _ in range(n_envs)]
        self.log_dir = log_dir
        self.n_envs = n_envs
        ctx = multiprocessing.get_context('spawn')
        self.queue = ctx.Queue()
        self.proc = ctx.Process(target=self.loop)
        self.proc.start()

    def log(self, env_rewards, predicted_rewards, dones):
        env_rewards = np.array(env_rewards)
        predicted_rewards = np.array(predicted_rewards)
        dones = np.array(dones)
        assert env_rewards.shape == (self.n_steps, self.n_envs)
        assert predicted_rewards.shape == (self.n_steps, self.n_envs)
        assert dones.shape == (self.n_steps, self.n_envs)
        for env_n in range(self.n_envs):
            env_rewards_n = env_rewards[:, env_n]
            predicted_rewards_n = predicted_rewards[:, env_n]
            dones_n = dones[:, env_n]
            for n in range(len(predicted_rewards_n)):
                env_reward = env_rewards_n[n]
                predicted_reward = predicted_rewards_n[n]
                self.env_rewards_by_env_n[env_n].append(env_reward)
                self.predicted_rewards_by_env_n[env_n].append(predicted_reward)
                if dones_n[n]:
                    self.queue.put(Rewards(env_n=env_n,
                                           env_rewards=self.env_rewards_by_env_n[env_n],
                                           predicted_rewards=self.predicted_rewards_by_env_n[env_n]))
                    self.logger.logkv('processed_reward', sum(self.predicted_rewards_by_env_n[env_n]))
                    self.predicted_rewards_by_env_n[env_n] = []

    def loop(self):
        self.episode_n = [0 for _ in range(self.n_envs)]

        log_files = []
        for env_n in range(self.n_envs):
            f = open(os.path.join(self.log_dir, f'reward_log_env_{env_n}.log'), 'w')
            log_files.append(f)

        image_dir = os.path.join(self.log_dir, 'train_env_rewards')
        os.makedirs(image_dir)

        figure()

        while True:
            rewards = self.queue.get()  # type: Rewards
            f = log_files[rewards.env_n]
            f.write(f'\nEpisode {self.episode_n[rewards.env_n]}\n')
            assert len(rewards.predicted_rewards) == len(rewards.env_rewards)
            for n in range(len(rewards.predicted_rewards)):
                f.write(f'{rewards.env_rewards[n]} {rewards.predicted_rewards[n]}\n')
            f.write('\n')

            true_rewards = rewards.env_rewards
            predicted_rewards = rewards.predicted_rewards
            predicted_rewards_rescaled = np.copy(predicted_rewards)
            predicted_rewards_rescaled -= np.min(predicted_rewards_rescaled)
            predicted_rewards_rescaled /= np.max(predicted_rewards_rescaled)
            predicted_rewards_rescaled *= (np.max(true_rewards) - np.min(true_rewards))
            predicted_rewards_rescaled -= np.min(true_rewards)

            clf()
            plot(rewards.predicted_rewards, label='Predicted rewards')
            plot(predicted_rewards_rescaled, label='Predicted rewards (rescaled)')
            plot(rewards.env_rewards, label='Environment rewards')

            savefig(os.path.join(image_dir, 'env_{}_episode_{}.png'.format(rewards.env_n,
                                                                           self.episode_n[rewards.env_n])))

            self.episode_n[rewards.env_n] += 1
