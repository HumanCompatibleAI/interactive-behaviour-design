from collections import deque
from functools import partial

import gym
import numpy as np
from gym.core import ObservationWrapper
from gym.envs import register as gym_register
from gym.envs.robotics import FetchReachEnv
from gym.spaces import Discrete
from gym.wrappers import FlattenDictWrapper

from wrappers.util_wrappers import CollectEpisodeStats, RepeatActions, LimitActions, CheckActionLimit


def decode_fetch_reach_obs(obs):
    obs_by_name = {
        'grip_pos': obs[:3],
        'gripper_state': obs[3:5],
        'grip_velp': obs[5:8],
        'grip_vel': obs[8:10],
        'goal_pos': obs[10:13],
    }
    obs_by_name['goal_rel_grip'] = obs_by_name['goal_pos'] - obs_by_name['grip_pos']
    return obs_by_name


class FetchReachObsWrapper(ObservationWrapper):
    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        obs = self.observation(np.zeros(self.env.observation_space.shape))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

    def observation(self, orig_obs):
        obs_dict = decode_fetch_reach_obs(orig_obs)
        obs = np.concatenate([obs_dict['grip_pos'],
                              obs_dict['goal_pos'],
                              obs_dict['goal_rel_grip']])
        return obs

    @staticmethod
    def decode(obs):
        obs_by_name = {
            'grip_pos': obs[:3],
            'goal_pos': obs[3:6],
            'goal_rel_grip': obs[6:9]
        }
        return obs_by_name


class FetchReachStatsWrapper(CollectEpisodeStats):
    # Should be matched to number of TD3 test episodes
    LOG_RATE_EVERY_N_EPISODES = 10

    def __init__(self, env):
        assert isinstance(env, FlattenDictWrapper)
        assert env.observation_space.shape == (13,), env.observation_space.shape
        CollectEpisodeStats.__init__(self, env)
        self.stats = {}
        self.last_stats = {}
        self.successes = []
        self.partial_success = False
        self.partial_successes = []
        self.successes_near_end = []
        self.gripper_to_goal_dist_list = deque(maxlen=10)
        self.last_obs = None
        self.n_steps = None

    def reset(self):
        if self.last_obs is not None:
            # We do this here rather than in step because we apply FetchStatsWrapper as part of the registered
            # environment and therefore before TimeLimit, so we never see done in step
            obs_by_name = decode_fetch_reach_obs(self.last_obs)
            gripper_to_goal_distance = np.linalg.norm(obs_by_name['goal_rel_grip'])

            success = (gripper_to_goal_distance < 0.05)
            self.stats['success'] = float(success)
            self.successes.append(success)
            if len(self.successes) == self.LOG_RATE_EVERY_N_EPISODES:
                self.stats['success_rate'] = self.successes.count(True) / len(self.successes)
                self.successes = []

            self.stats['success_partial'] = float(self.partial_success)
            self.partial_successes.append(self.partial_success)
            if len(self.partial_successes) == self.LOG_RATE_EVERY_N_EPISODES:
                self.stats['success_partial_rate'] = self.partial_successes.count(True) / len(self.partial_successes)
                self.partial_successes = []

            success_near_end = any([d < 0.05 for d in self.gripper_to_goal_dist_list])
            self.stats['success_near_end'] = float(success_near_end)
            self.successes_near_end.append(success_near_end)
            if len(self.successes_near_end) == self.LOG_RATE_EVERY_N_EPISODES:
                self.stats['success_near_end_rate'] = self.successes_near_end.count(True) / len(self.successes_near_end)
                self.successes_near_end = []

            avg = self.stats['gripper_to_target_cumulative_distance'] / self.n_steps
            self.stats['gripper_to_target_average_distance'] = avg

        self.last_stats = dict(self.stats)
        self.stats['gripper_to_target_cumulative_distance'] = 0
        self.partial_success = False
        self.last_obs = None
        self.n_steps = 0

        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_by_name = decode_fetch_reach_obs(obs)
        gripper_to_goal_distance = np.linalg.norm(obs_by_name['goal_rel_grip'])
        self.stats['gripper_to_target_cumulative_distance'] += gripper_to_goal_distance
        self.gripper_to_goal_dist_list.append(gripper_to_goal_distance)
        if gripper_to_goal_distance < 0.05:
            self.partial_success = True
        self.last_obs = obs
        self.n_steps += 1
        return obs, reward, done, info


def make_env(action_repeat, action_limit):
    env = FetchReachEnv(reward_type='dense')
    env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env = FetchReachStatsWrapper(env)
    env = FetchReachObsWrapper(env)
    env = RepeatActions(env, action_repeat)
    env = CheckActionLimit(env, action_limit)
    env = LimitActions(env, action_limit)
    return env


def register():
    for action_limit in [0.2, 1]:
        for action_repeat in [1, 5]:
            gym_register(f'FetchReach-CustomActionRepeat{action_repeat}ActionLimit{action_limit}-v0',
                         entry_point=partial(make_env, action_repeat=action_repeat, action_limit=action_limit),
                         max_episode_steps=250)
