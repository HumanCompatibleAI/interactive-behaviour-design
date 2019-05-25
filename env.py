import multiprocessing
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
    SaveEpisodeStats, LogEpisodeStats, DummyRender


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


def make_env(env_id, num_env, seed, experience_dir,
             reset_state_server_queue, reset_mode_value,
             reset_state_receiver_queue: multiprocessing.Queue,
             episode_obs_queue: multiprocessing.Queue,
             segments_queue: multiprocessing.Queue,
             render_segments,
             render_every_nth_episode,
             save_states):
    def make_env_fn(rank):
        def _thunk():
            np.random.seed(seed + rank)
            random.seed(seed + rank)

            env = gym.make(env_id)
            if isinstance(env.env, AtariEnv):
                env = make_atari_env_with_preprocessing(env_id)
            env.seed(seed + rank)
            set_timeouts(env)

            # needs to be done before preprocessing
            unwrapped_env = env.unwrapped
            first_wrapper = unwrap_to(env, type(unwrapped_env), n_before=1)
            first_wrapper.env = SaveEpisodeStats(unwrapped_env)
            env = SaveEpisodeStats(env, rewards_only=True, suffix='_post_wrappers')

            env = StateBoundaryWrapper(env)

            env = StoredResetsWrapper(env, reset_mode_value, reset_state_server_queue)
            if rank == 0:
                env = LogEpisodeStats(env, suffix='_train', log_dir=experience_dir)
                env = Monitor(env, experience_dir,
                              lambda n: n and n % render_every_nth_episode == 0)  # save videos
                env = SaveEpisodeObs(env, episode_obs_queue)
                if global_variables.segment_save_mode == 'single_env':
                    if not render_segments:
                        env = DummyRender(env)
                    env = SaveSegments(env, segments_queue)
                if save_states:
                    # This is slow, so we disable it sometimes
                    env = SaveMidStateWrapper(env, reset_state_receiver_queue)

            if global_variables.segment_save_mode == 'multi_env' and not render_segments:
                # Segments are saved by VecSaveSegments after SubprocVecEnv
                env = DummyRender(env)

            return env

        return _thunk

    return CustomSubprocVecEnv([make_env_fn(i) for i in range(num_env)])
