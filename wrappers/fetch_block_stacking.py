from collections import deque
from functools import partial

import gym
import numpy as np
from gym.core import Wrapper, ObservationWrapper, ActionWrapper
from gym.envs import register as gym_register
from gym.spaces import Discrete
from gym.wrappers import FlattenDictWrapper

from fetch_block_stacking_env.env import FetchBlockStackingEnv
from utils import RunningProportion
from wrappers.util_wrappers import CollectEpisodeStats, RepeatActions

"""
Put object0 (black) on object1 (red).
"""


def decode_obs(obs):
    obs_by_name = {
        'grip_pos': obs[:3],
        'gripper_state': obs[3:5],
        'object0_pos': obs[5:8],
        'object0_rel_pos': obs[8:11],
        'object1_pos': obs[11:14],
        'object1_rel_pos': obs[14:17],
    }
    block_height = 0.05
    target_pos = obs_by_name['object1_pos'] + [0, 0, block_height]
    obs_by_name.update({
        'target_pos': target_pos,
        'object0_rel_target_pos': target_pos - obs_by_name['object0_pos']
    })
    return obs_by_name


class FetchBlockStackingObsWrapper(ObservationWrapper):
    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        obs = self.observation(np.zeros(self.env.observation_space.shape))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

    def observation(self, orig_obs):
        obs_dict = decode_obs(orig_obs)
        obs = np.concatenate([obs_dict['object0_rel_pos'],
                              obs_dict['object0_rel_target_pos'],
                              # The gripper is supposed to be symmetrical.
                              # But if the grippers are closed on the block and the arm is dragging the
                              # block across the table, both grippers can slightly translate in the
                              # MuJoCo simulation. So we need to look at both gripper positions to get the
                              # actual width.
                              [np.sum(obs_dict['gripper_state'])]])
        return obs

    @staticmethod
    def decode(obs):
        obs_by_name = {
            'object0_rel_pos': obs[:3],
            'object0_rel_target_pos': obs[3:6],
            'gripper_state': obs[6],
        }
        return obs_by_name


class FetchBlockStackingRewardWrapper(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.last_grip_width = None
        self.last_obs_by_name = None

    @staticmethod
    def _r(d):
        r = d ** 2
        r += 0.1 * np.log(d + 1e-3)  # maxes out at -0.69
        return -0.7 - r  # maxes out at 0.0

    @classmethod
    def _reward(cls, obs_by_name):
        d1 = np.linalg.norm(obs_by_name['object0_rel_pos'])
        d2 = np.linalg.norm(obs_by_name['object0_rel_target_pos'])
        reward = cls._r(d1) + cls._r(d2)
        return reward

    @staticmethod
    def object_between_grippers(obs_by_name):
        return all([abs(d) < 0.02 for d in obs_by_name['object0_rel_pos']])

    def step(self, action, return_decoded_obs=False):
        obs, reward_orig, done, info = self.env.step(action)
        obs_by_name = decode_obs(obs)
        reward = self._reward(obs_by_name)
        grip_width = np.sum(obs_by_name['gripper_state'])
        # maxes out at +0.5 reward
        reward += 5 * grip_width
        if self.object_between_grippers(obs_by_name):
            # about +0.5 reward for fully closed around block
            reward += 15 * (0.1 - grip_width)
        return obs, reward, done, info

    def reset(self):
        self.last_obs_by_name = None
        return self.env.reset()


class BinaryGripperWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.closed = False

    def action(self, action):
        if action[3] > 0.1:
            self.closed = False
        elif action[3] < -0.1:
            self.closed = True
        if self.closed:
            action[3] = -1
        else:
            action[3] = 1
        return action


def make_env(binary_gripper, action_repeat):
    env = FetchBlockStackingEnv()
    env = FlattenDictWrapper(env, ['observation'])
    env = FetchBlockStackingStatsWrapper(env)
    env = FetchBlockStackingRewardWrapper(env)
    if binary_gripper:
        env = BinaryGripperWrapper(env)
    env = FetchBlockStackingObsWrapper(env)
    env = RepeatActions(env, action_repeat)
    return env


def register():
    for action_repeat in [1, 5]:
        gym_register(f'FetchBlockStackingDenseRepeat{action_repeat}-v0',
                     entry_point=partial(make_env, binary_gripper=False, action_repeat=action_repeat),
                     max_episode_steps=250)
        gym_register(f'FetchBlockStackingDenseRepeat{action_repeat}BinaryGripper-v0',
                     entry_point=partial(make_env, binary_gripper=True, action_repeat=action_repeat),
                     max_episode_steps=250)


class FetchBlockStackingStatsWrapper(CollectEpisodeStats):
    # Should be matched to number of TD3 test episodes
    LOG_RATE_EVERY_N_EPISODES = 10
    SUCCESS_DISTANCE = 0.05

    def __init__(self, env):
        assert isinstance(env, FlattenDictWrapper)
        assert env.observation_space.shape == (17,)
        super().__init__(env)
        self.stats = {}
        self.last_stats = {}
        self.aligned_proportion = None
        self.gripping_proportion = None
        self.successes = []
        self.partial_success = False
        self.partial_successes = []
        self.object0_to_target_distance_list = deque(maxlen=10)
        self.successes_near_end = []
        self.last_obs = None
        self.n_steps = None

    @staticmethod
    def check_success(obs_by_name):
        object0_to_target_dist = np.linalg.norm(obs_by_name['object0_rel_target_pos'])
        success = (object0_to_target_dist < FetchBlockStackingStatsWrapper.SUCCESS_DISTANCE)
        return success

    def reset(self):
        if self.last_obs is not None:
            # We do this here rather than in step because we apply FetchStatsWrapper as part of the registered
            # environment and therefore before TimeLimit, so we never see done in step
            obs_by_name = decode_obs(self.last_obs)

            success = self.check_success(obs_by_name)
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

            success_near_end = any([d < self.SUCCESS_DISTANCE for d in self.object0_to_target_distance_list])
            self.stats['success_near_end'] = float(success_near_end)
            self.successes_near_end.append(success_near_end)
            if len(self.successes_near_end) == self.LOG_RATE_EVERY_N_EPISODES:
                self.stats['success_near_end_rate'] = self.successes_near_end.count(True) / len(self.successes_near_end)
                self.successes_near_end = []

            avg_dist = self.stats['object0_to_target_cumulative_distance'] / self.n_steps
            self.stats['object0_to_target_average_distance'] = avg_dist

        self.last_stats = dict(self.stats)
        self.stats['ep_frac_aligned_with_object0'] = 0
        self.stats['ep_frac_gripping_object0'] = 0
        self.stats['gripper_to_object0_cumulative_distance'] = 0
        self.stats['object0_to_target_cumulative_distance'] = 0
        self.stats['object0_to_target_min_distance'] = float('inf')
        self.aligned_proportion = RunningProportion()
        self.gripping_proportion = RunningProportion()
        self.partial_success = False
        self.last_obs = None
        self.n_steps = 0

        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs

        obs_by_name = decode_obs(obs)

        self.stats['gripper_to_object0_cumulative_distance'] += np.linalg.norm(obs_by_name['object0_rel_pos'])

        object0_to_target_distance = np.linalg.norm(obs_by_name['object0_rel_target_pos'])
        self.stats['object0_to_target_cumulative_distance'] += object0_to_target_distance

        if object0_to_target_distance < self.stats['object0_to_target_min_distance']:
            self.stats['object0_to_target_min_distance'] = object0_to_target_distance

        aligned_with_object0 = (np.linalg.norm(obs_by_name['object0_rel_pos']) < 0.04)
        self.aligned_proportion.update(float(aligned_with_object0))
        self.stats['ep_frac_aligned_with_object0'] = self.aligned_proportion.v

        grippers_closed = np.sum(obs_by_name['gripper_state']) < 0.05
        gripping_object0 = aligned_with_object0 and grippers_closed
        self.gripping_proportion.update(float(gripping_object0))
        self.stats['ep_frac_gripping_object0'] = self.gripping_proportion.v

        self.object0_to_target_distance_list.append(object0_to_target_distance)

        if self.check_success(obs_by_name):
            self.partial_success = True

        self.n_steps += 1

        return obs, reward, done, info
