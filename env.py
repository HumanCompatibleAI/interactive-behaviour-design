import multiprocessing
import os
import random

import numpy as np
from gym.envs.atari import AtariEnv
from gym.envs.robotics import FetchEnv
from gym.wrappers import Monitor, TimeLimit

import global_variables
from a2c.common import gym
from global_constants import ROLLOUT_FPS
from subproc_vec_env_custom import CustomSubprocVecEnv
from utils import unwrap_to
from wrappers.atari_generic import make_atari_env_with_preprocessing
from wrappers.lunar_lander_stateful import LunarLanderStateful
from wrappers.state_boundary_wrapper import StateBoundaryWrapper
from wrappers.util_wrappers import StoredResetsWrapper, SaveMidStateWrapper, SaveEpisodeObs, SaveSegments, \
    CollectEpisodeStats, SaveEpisodeStats, DummyRender


def set_timeouts(env):
    assert isinstance(env, TimeLimit), env

    # Needed to prevent random resets in the demonstration environment
    env._max_episode_seconds = None

    if isinstance(env.unwrapped, FetchEnv):
        if 'Repeat1' in env.unwrapped.spec.id:
            max_seconds = 10
        elif 'Repeat3' in env.unwrapped.spec.id:
            max_seconds = 5
        else:
            raise Exception()
    elif isinstance(env.unwrapped, AtariEnv):
        max_minutes = 5
        max_seconds = max_minutes * 60
    elif isinstance(env.unwrapped, LunarLanderStateful):
        max_seconds = 20
    else:
        raise Exception()

    env._max_episode_steps = ROLLOUT_FPS * max_seconds


def make_envs(env_id, num_env, seed, log_dir,
              reset_state_server_queue, reset_mode_value,
              reset_state_receiver_queue: multiprocessing.Queue,
              episode_obs_queue: multiprocessing.Queue,
              segments_queue: multiprocessing.Queue,
              render_segments,
              render_every_nth_episode,
              save_states):
    def make_env_fn(env_type, env_n=0):

        def _thunk():
            assert env_type in ['train', 'test', 'demo']

            if env_type == 'train':
                # Only do this for SubprocVecEnv environments
                # Otherwise we'll reset the seed of the main process
                np.random.seed(seed + env_n)
                random.seed(seed + env_n)

            # For all env_types
            env = gym.make(env_id)
            if isinstance(env.unwrapped, AtariEnv):
                env = make_atari_env_with_preprocessing(env_id)
            env.seed(seed + env_n)
            set_timeouts(env)
            # Save stats from before preprocessing
            unwrapped_env = env.unwrapped
            first_wrapper = unwrap_to(env, type(unwrapped_env), n_before=1)
            first_wrapper.env = CollectEpisodeStats(unwrapped_env)
            # Also save stats from after preprocessing
            env = CollectEpisodeStats(env, rewards_only=True, suffix='_after_preprocessing')

            if env_type == 'train':
                env = StateBoundaryWrapper(env)
                env = StoredResetsWrapper(env, reset_mode_value, reset_state_server_queue)

            if env_type == 'train' and env_n == 0:
                train_log_dir = os.path.join(log_dir, 'train_env')
                env = SaveEpisodeStats(env, suffix='_train', log_dir=train_log_dir)
                env = Monitor(env, train_log_dir, lambda n: n and n % render_every_nth_episode == 0)  # Save videos
                env = SaveEpisodeObs(env, episode_obs_queue)  # For labelling of frames for classifiers
                if global_variables.segment_save_mode == 'single_env':
                    if not render_segments:
                        env = DummyRender(env)
                    env = SaveSegments(env, segments_queue)
                elif global_variables.segment_save_mode == 'multi_env':
                    # Segments are saved by VecSaveSegments after SubprocVecEnv
                    if not render_segments:
                        env = DummyRender(env)
                if save_states:
                    env = SaveMidStateWrapper(env, reset_state_receiver_queue)

            if env_type == 'test':
                test_log_dir = os.path.join(log_dir, 'test_env')
                env = SaveEpisodeStats(env, suffix='_test', log_dir=test_log_dir)
                env = Monitor(env, test_log_dir, lambda n: True)

            return env

        return _thunk

    train_env = CustomSubprocVecEnv([make_env_fn('train', i) for i in range(num_env)])
    test_env = make_env_fn('test')()
    demo_env = make_env_fn('demo')()

    return train_env, test_env, demo_env
