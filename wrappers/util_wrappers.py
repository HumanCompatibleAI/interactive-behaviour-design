import logging
import multiprocessing
import os
import pickle
import queue
import tempfile
import time
from collections import deque
from enum import Enum
from functools import partial
from multiprocessing import Queue

from gym.wrappers import TimeLimit
from os import path as osp
from threading import Thread, Lock
from typing import Dict

import cv2
import easy_tf_log
import numpy as np
from gym.core import ObservationWrapper, Wrapper

from baselines import logger
from a2c.common.vec_env import VecEnvWrapper
from classifier_collection import ClassifierCollection
from drlhp.reward_predictor import RewardPredictor
from utils import unwrap_to, EnvState, TimerContext


class DrawClassifierPredictionWrapper(ObservationWrapper):
    classifier: ClassifierCollection

    def __init__(self, env, classifier):
        ObservationWrapper.__init__(self, env)
        self.classifier = classifier

    def observation(self, obs):
        obs = np.copy(obs)
        obs_width = obs.shape[1]
        labels = self.classifier.get_labels()
        for label_n, label_name in enumerate(labels):
            pred = self.classifier.predict(label_name, [obs])[0]
            if pred == 0:
                color = (0, 0, 0, 0)
            elif pred == 1:
                color = (255, 255, 255, 255)
            else:
                raise Exception("Bad pred: '{}'".format(pred))
            x = int(label_n * obs_width / len(labels))
            cv2.putText(obs,
                        label_name,
                        org=(5 + x, 20),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1.0,
                        color=color,
                        thickness=2)
        return obs


class RewardSource(Enum):
    CLASSIFIER = 0
    DRLHP = 1
    ENV = 2
    NONE = 3


class VecLogRewards(VecEnvWrapper):
    def __init__(self, venv, log_dir, postfix=None):
        VecEnvWrapper.__init__(self, venv)
        self.episode_reward_sum = 0
        self.logger = easy_tf_log.Logger()
        self.logger.set_log_dir(log_dir)
        self.log_key = 'env/episode_reward_sum_vec'
        if postfix is not None:
            self.log_key += postfix

    def step_wait(self):
        obses, rewards, dones, infos = self.venv.step_wait()
        self.episode_reward_sum += rewards[0]
        if dones[0]:
            self.logger.logkv(self.log_key, self.episode_reward_sum)
            self.episode_reward_sum = 0
        return obses, rewards, dones, infos

    def reset(self):
        return self.venv.reset()


class VecRewardSwitcherWrapper(VecEnvWrapper):

    def __init__(self, venv, classifiers: ClassifierCollection,
                 network, network_args, reward_predictor_std, log_dir):
        VecEnvWrapper.__init__(self, venv)

        drlhp_log_dir = os.path.join(log_dir, 'drlhp')
        obs_shape = venv.observation_space.shape
        self.reward_predictor = RewardPredictor(network=network, network_args=network_args,
                                                log_dir=drlhp_log_dir, obs_shape=obs_shape,
                                                r_std=reward_predictor_std)

        self.classifiers = classifiers
        self.cur_classifier_name = None
        self.cur_reward_source = RewardSource.NONE

    def set_reward_source(self, reward_source: RewardSource):
        self.cur_reward_source = reward_source

    def reward_drlhp(self, obs):
        rewards = self.reward_predictor.reward(obs)
        assert rewards.shape == (self.venv.num_envs,)
        return rewards

    def reward_classifier(self, obs):
        if self.cur_classifier_name is None:
            print("Warning: classifier not set")
            return [0.0] * len(obs)

        probs = self.classifiers.predict_positive_prob(self.cur_classifier_name, obs)
        assert probs.shape == (self.venv.num_envs,)
        rewards = (probs >= 0.5).astype(np.float32)
        return rewards

    def step_wait(self):
        obs, rewards_env, dones, infos = self.venv.step_wait()
        assert obs.shape[0] == self.venv.num_envs
        if self.cur_reward_source == RewardSource.ENV:
            rewards = rewards_env
        elif self.cur_reward_source == RewardSource.CLASSIFIER:
            rewards = self.reward_classifier(obs)
        elif self.cur_reward_source == RewardSource.DRLHP:
            rewards = self.reward_drlhp(obs)
        elif self.cur_reward_source == RewardSource.NONE:
            rewards = [0.0] * self.venv.num_envs
        return obs, rewards, dones, infos

    def reset(self):
        return self.venv.reset()


class LogEpisodeStats(Wrapper):
    def __init__(self, env, log_dir=None, suffix=None, stdout=False):
        Wrapper.__init__(self, env)
        self.stdout = stdout
        if log_dir is None:
            self.logger = None
        else:
            self.logger = easy_tf_log.Logger(log_dir)
        self.suffix = ('' if suffix is None else suffix)
        self.stats_envs = None
        self.stats = {}
        self.set_env(env)

    def set_env(self, env):
        self.env = env
        self.stats_envs = []
        while True:
            if isinstance(env, SaveEpisodeStats):
                self.stats_envs.append(env)
            try:
                env = env.env
            except:
                break

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = self.env.reset()
        self.stats = {}
        for stats_env in self.stats_envs:
            for key, value in stats_env.last_stats.items():
                self.stats[key] = value
                key_with_suffix = f'env{self.suffix}/' + key
                if self.logger:
                    self.logger.logkv(key_with_suffix, value)
                if self.stdout:
                    print(key_with_suffix + ':', value)
            stats_env.last_stats = {}
        return obs


class LogDoneInfo(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.logger = easy_tf_log.Logger()
        self.logger.set_log_dir(osp.join(logger.get_dir(), "env2"))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            for key, value in info.items():
                self.logger.logkv('env/' + key, value)
        return obs, reward, done, info


class SaveEpisodeStats(Wrapper):
    """
    Save per-episode rewards and episode lengths.
    """

    def __init__(self, env, suffix='', rewards_only=False):
        Wrapper.__init__(self, env)
        self.suffix = suffix
        self.rewards_only = rewards_only
        self.episode_n = -1
        self.episode_rewards = self.episode_length_steps = None
        self.last_stats = {}

    def reset(self):
        self.last_stats = {}
        if self.episode_n >= 0:
            self.last_stats['reward_sum' + self.suffix] = sum(self.episode_rewards)
            if not self.rewards_only:
                self.last_stats['episode_n' + self.suffix] = self.episode_n
                self.last_stats['episode_length_steps' + self.suffix] = self.episode_length_steps

        self.episode_n += 1
        self.episode_rewards = []
        self.episode_length_steps = 0

        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_rewards.append(reward)
        self.episode_length_steps += 1
        return obs, reward, done, info


class SeaquestStatsWrapper(SaveEpisodeStats):
    def __init__(self, env):
        SaveEpisodeStats.__init__(self, env)
        self.last_n_divers = None
        self.n_diver_pickups = None

    def reset(self):
        self.last_stats = {}
        if self.n_diver_pickups is not None:
            self.last_stats['n_diver_pickups'] = self.n_diver_pickups
        self.last_n_divers = 0
        self.n_diver_pickups = 0
        return self.env.reset()

    def get_n_divers_picked_up(self):
        return self.env.unwrapped.ale.getRAM()[62]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        cur_n_divers = self.get_n_divers_picked_up()
        if cur_n_divers > self.last_n_divers:
            self.n_diver_pickups += (cur_n_divers - self.last_n_divers)
        self.last_n_divers = cur_n_divers

        return obs, reward, done, info


class ResetMode(Enum):
    USE_ENV_RESET = 0
    FROM_STATE_CACHE = 1


class QueueEndpoint(Enum):
    DEMONSTRATIONS = 0
    TRAINING = 1


class ResetStateCache:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.logger = easy_tf_log.Logger()
        self.logger.set_log_dir(os.path.join(log_dir, 'reset_states'))
        self.in_queues = dict()  # type: Dict[QueueEndpoint, multiprocessing.Queue]
        self.out_queues = dict()  # type: Dict[QueueEndpoint, multiprocessing.Queue]
        self.receive_to_pool = dict()  # type: Dict[QueueEndpoint, str]
        self.serve_from_pool = dict()  # type: Dict[QueueEndpoint, str]
        self.pools = dict()  # type: Dict[str, deque]
        self.lock = Lock()
        for endpoint in [QueueEndpoint.DEMONSTRATIONS, QueueEndpoint.TRAINING]:
            # We limit the size so that we don't get a massive backlog
            self.in_queues[endpoint] = multiprocessing.Queue(maxsize=50)
            self.out_queues[endpoint] = multiprocessing.Queue(maxsize=1)
            self.receive_to_pool[endpoint] = None
            self.serve_from_pool[endpoint] = None
        self.queue_from_demonstrations = self.in_queues[QueueEndpoint.DEMONSTRATIONS]
        self.queue_from_training = self.in_queues[QueueEndpoint.TRAINING]
        self.queue_to_demonstrations = self.out_queues[QueueEndpoint.DEMONSTRATIONS]
        self.queue_to_training = self.out_queues[QueueEndpoint.TRAINING]

    def start_saver_receiver(self):
        for endpoint in [QueueEndpoint.DEMONSTRATIONS, QueueEndpoint.TRAINING]:
            Thread(target=self.in_queue_receiver, args=[endpoint]).start()
            Thread(target=self.out_queue_server, args=[endpoint]).start()

    def in_queue_receiver(self, endpoint: QueueEndpoint):
        queue = self.in_queues[endpoint]
        while True:
            reset_state = queue.get()
            pool_name = self.receive_to_pool[endpoint]
            if pool_name is None:
                continue
            if pool_name not in self.pools:
                print("Warning: pool '{}' not found".format(pool_name))
                continue
            self.lock.acquire()
            self.pools[pool_name].append(reset_state)
            self.lock.release()
            self.logger.logkv(f'reset_states/{pool_name}_len', len(self.pools[pool_name]))

    def out_queue_server(self, target: QueueEndpoint):
        queue = self.out_queues[target]
        while True:
            pool_name = self.serve_from_pool[target]
            if pool_name is None:
                time.sleep(5.0)
                continue
            if pool_name not in self.pools:
                print("Warning: pool '{}' not found".format(pool_name))
                time.sleep(5.0)
                continue
            pool = self.pools[pool_name]
            if len(pool) == 0:
                print("Warning: pool '{}' is empty".format(pool_name))
                time.sleep(5.0)
                continue
            n = np.random.choice(len(pool))
            reset_state = pool[n]
            queue.put(reset_state)
            msg_str = "Supplied reset state {} to {} from pool '{}'".format(
                n, str(target), pool_name)
            print(msg_str)

    def add_pool(self, name, max_len):
        if name in self.pools:
            raise Exception("Pool '{}' already exists".format(name))
        self.pools[name] = deque(maxlen=max_len)

    def load_dir(self, dir):
        path = os.path.join(dir, 'reset_states.pkl')
        if not os.path.exists(path):
            print("Warning: reset_states.pkl not found")
            return
        with open(path, 'rb') as f:
            self.pools = pickle.load(f)
        print("Loaded reset states from '{}'".format(path))

    def save(self):
        self.lock.acquire()
        fd, temp_file = tempfile.mkstemp(dir=self.log_dir)
        with open(temp_file, 'wb') as f:
            pickle.dump(self.pools, f)
        os.close(fd)
        save_path = os.path.join(self.log_dir, 'reset_states.pkl')
        os.rename(temp_file, save_path)
        self.lock.release()


class StoredResetsWrapper(Wrapper):

    def __init__(self, env, reset_mode_value: multiprocessing.Value, reset_state_queue: Queue):
        Wrapper.__init__(self, env)
        self.reset_state_queue = reset_state_queue
        self.mode = reset_mode_value
        self.env_boundary = unwrap_to(self.env, StateBoundaryWrapper)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        if self.mode.value == ResetMode.USE_ENV_RESET.value:
            return self.env.reset()
        elif self.mode.value == ResetMode.FROM_STATE_CACHE.value:
            reset_state = self.get_reset_state()
            self.env_boundary.env = reset_state.env
            return reset_state.obs
        else:
            print("Warning: unknown reset mode '{}'; using env reset".format(self.mode.value))
            return self.env.reset()

    def get_reset_state(self):
        while True:
            try:
                reset_state = self.reset_state_queue.get(block=True,
                                                         timeout=0.1)  # type: EnvState
                break
            except queue.Empty:
                print("Waiting for reset state...")
                time.sleep(1.0)
        return reset_state


class SaveMidStateWrapper(Wrapper):
    SAVE_EVERY_NTH_STEP = 10

    def __init__(self, env, state_save_queue: multiprocessing.Queue, verbose=True):
        Wrapper.__init__(self, env)
        self.state_save_queue = state_save_queue
        self.verbose = verbose
        self.states = None
        self.step_n = 1
        self.states = []
        self.env_boundary = unwrap_to(self.env, StateBoundaryWrapper)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Make sure we save at least one step per episode
        if self.step_n == 1 or self.step_n % SaveMidStateWrapper.SAVE_EVERY_NTH_STEP == 0:
            reset_state = EnvState(self.env_boundary.env, obs, done, self.step_n)
            self.states.append(reset_state)
        self.step_n += 1
        return obs, reward, done, info

    def filter_states(self):
        if 'Seaquest' in self.env.spec.id:
            # Filter out states which are too close to the end
            # (after the submarine has already died)
            self.states = [state for state in self.states if state.step_n < self.step_n - 25]

    def reset(self):
        self.filter_states()

        if self.states:
            state_n = np.random.randint(low=0, high=len(self.states))
            state = self.states[state_n]
            try:
                self.state_save_queue.put(state, block=False)
                if self.verbose:
                    print("Saved reset state from step", state.step_n)
            except queue.Full:
                pass

        self.states = []
        self.step_n = 1

        return self.env.reset()


class StateBoundaryWrapper(Wrapper):
    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class OneLifeWrapper(Wrapper):

    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.last_lives = None

    def reset(self):
        self.last_lives = self.env.unwrapped.ale.lives()
        return self.env.reset()

    def step(self, a):
        obs, reward, done, info = self.env.step(a)
        if self.env.unwrapped.ale.lives() < self.last_lives:
            done = True
        self.last_lives = self.env.unwrapped.ale.lives()
        return obs, reward, done, info


class SaveEpisodeObs(Wrapper):
    def __init__(self, env, queue):
        Wrapper.__init__(self, env)
        self.queue = queue
        self.obses = []
        self.episode_n = -1

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obses.append(obs)
        return obs, reward, done, info

    def send_frames(self):
        if self.obses:
            self.queue.put((self.episode_n, self.obses))

    def reset(self):
        self.send_frames()
        self.episode_n += 1
        obs = self.env.reset()
        self.obses = [obs]
        return obs


class DummyRender(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env)

    def render(self, mode='rgb_array', **kwargs):
        assert mode == 'rgb_array'
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()


class SaveSegments(Wrapper):
    FRAMES_PER_SEGMENT = 30

    def __init__(self, env, segment_queue: multiprocessing.Queue):
        Wrapper.__init__(self, env)
        self.queue = segment_queue
        self.segment_frames = None
        self.segment_obses = None
        self.segment_rewards = None
        self.episode_n = -1
        self._reset_segment()

    def _reset_segment(self):
        self.segment_frames = []
        self.segment_obses = []
        self.segment_rewards = []

    def _pad_segment(self):
        while len(self.segment_obses) < self.FRAMES_PER_SEGMENT:
            self.segment_frames.append(self.segment_frames[-1])
            self.segment_obses.append(self.segment_obses[-1])
            self.segment_rewards.append(self.segment_rewards[-1])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.segment_frames.append(self.env.render(mode='rgb_array'))
        self.segment_obses.append(np.copy(obs))
        self.segment_rewards.append(reward)
        if done or len(self.segment_obses) == self.FRAMES_PER_SEGMENT:
            self._pad_segment()
            tuple = (self.segment_obses, self.segment_rewards, self.segment_frames, self.episode_n)
            try:
                self.queue.put(tuple, block=False)
            except queue.Full:
                pass
            self._reset_segment()
        return obs, reward, done, info

    def reset(self):
        # We assume we're operating in a SubprocVecEnv, which normally doesn't need to be reset, so that if we
        # receive an explicit reset, we're doing something unusual. We might be part-way through an episode and only
        # have a couple of frames in the segment so far, so let's play it safe by dropping the current segment.
        self._reset_segment()
        self.episode_n += 1
        return self.env.reset()


class EpisodeLengthLimitWrapper(Wrapper):
    env: TimeLimit

    def __init__(self, env, max_episode_steps_value: multiprocessing.Value):
        Wrapper.__init__(self, env)
        self.max_episode_steps_value = max_episode_steps_value
        self.n_steps = 0

    def reset(self):
        self.n_steps = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.n_steps += 1
        if self.n_steps > self.max_episode_steps_value.value:
            done = True
        return obs, reward, done, info


class Profile(Wrapper):
    @staticmethod
    def _time_step(step_fn, action):
        with TimerContext(str(step_fn)):
            obs, reward, done, info = step_fn(action)
        return obs, reward, done, info

    @staticmethod
    def _print_sep_and_step(step_fn, action):
        print()
        return step_fn(action)

    def __init__(self, env):
        Wrapper.__init__(self, env)
        while True:
            env.step = partial(self._time_step, env.step)
            try:
                env = env.env
            except:
                env.step = partial(self._print_sep_and_step, env.step)
                break

    def reset(self):
        return self.env.reset()


class RepeatActions(Wrapper):
    def __init__(self, env, repeat_n):
        Wrapper.__init__(self, env)
        assert repeat_n > 0
        self.repeat_n = repeat_n

    def step(self, action):
        rewards = []
        for _ in range(self.repeat_n):
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
            if done:
                break
        return obs, sum(rewards), done, info

    def reset(self):
        return self.env.reset()
