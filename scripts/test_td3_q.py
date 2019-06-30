#!/usr/bin/env python3

import argparse
import os
import sys
import tempfile
from collections import deque

import cv2
import gym
import numpy as np
from gym import Wrapper, ObservationWrapper
from gym.spaces import Box
from gym.utils.play import play
from gym.wrappers import TimeLimit

from wrappers.wrappers_debug import RewardGrapher

import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from policies.td3 import TD3Policy
from drlhp.reward_predictor import RewardPredictor
from drlhp.reward_predictor_core_network import net_cnn, net_mlp
from wrappers.seaquest_reward import register as seaquest_register
from wrappers.breakout_reward import register as breakout_register
from wrappers.fetch_reach import register as fetch_reach_register
from wrappers.fetch_block_stacking import register as fetch_bs_register
from wrappers.util_wrappers import FetchDiscreteActions


class DrawRewards(Wrapper):
    def __init__(self, env, reward_predictor: RewardPredictor):
        super().__init__(env)
        self.reward_predictor = reward_predictor
        self.grapher_true_reward = RewardGrapher(scale=3, y=20, name='True reward')
        self.grapher_predicted_reward = RewardGrapher(scale=2, y=150, name='Predicted reward')

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.grapher_true_reward.values.append(reward)
        predicted_reward = self.reward_predictor.normalized_rewards(obs, action)
        self.grapher_predicted_reward.values.append(predicted_reward)
        return obs, reward, done, info

    def render(self, mode='human', **kwargs):
        assert mode == 'rgb_array'
        frame = self.env.render(mode='rgb_array')
        frame = np.concatenate([np.zeros((300, 500, 3)), frame])
        self.grapher_true_reward.draw(frame)
        self.grapher_predicted_reward.draw(frame)
        return frame


class RenderObs(ObservationWrapper):
    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        im = self.env.render(mode='rgb_array')
        self.observation_space = Box(low=0, high=255, shape=im.shape)

    def observation(self, observation):
        return self.env.render(mode='rgb_array')


parser = argparse.ArgumentParser()
parser.add_argument('env_id')
parser.add_argument('td3_ckpt')
args = parser.parse_args()

seaquest_register()
breakout_register()
fetch_reach_register()
fetch_bs_register()
env = gym.make(args.env_id)  # type: TimeLimit
env._max_episode_seconds = None
env._max_episode_steps = None

td3 = TD3Policy('dummy_name', args.env_id, env.observation_space, env.action_space,
                n_envs=16)
td3.load_checkpoint(args.td3_ckpt)


class RewardPredictor:
    def __init__(self, td3: TD3Policy):
        self.td3 = td3

    def reward(self, obs, action):
        return self.td3.sess.run(self.td3.q1, feed_dict={self.td3.x_ph: [obs],
                                                         self.td3.a_ph: [action]})[0]


reward_predictor = RewardPredictor(td3)

env = DrawRewards(env, reward_predictor)
env = RenderObs(env)
if 'Fetch' in args.env_id:
    env = FetchDiscreteActions(env)

play(env, zoom=1, fps=30)
