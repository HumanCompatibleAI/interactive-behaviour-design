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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from drlhp.reward_predictor import RewardPredictor
from drlhp.reward_predictor_core_network import net_cnn, net_mlp
from wrappers.seaquest_reward import register as seaquest_register
from wrappers.breakout_reward import register as breakout_register
from wrappers.fetch_reach import register as fetch_reach_register
from wrappers.util_wrappers import FetchDiscreteActions


class DrawRewards(Wrapper):
    def __init__(self, env, reward_predictor: RewardPredictor):
        super().__init__(env)
        self.reward_predictor = reward_predictor
        self.grapher_true_reward = RewardGrapher(scale=1, y=20)
        self.grapher_predicted_reward = RewardGrapher(scale=1, y=150)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.grapher_true_reward.values.append(reward)
        predicted_reward = reward_predictor.raw_rewards(np.array([obs]))[0][0]
        self.grapher_predicted_reward.values.append(predicted_reward)
        return obs, reward, done, info

    def render(self, mode='human', **kwargs):
        assert mode == 'rgb_array'
        frame = self.env.render(mode='rgb_array')
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
parser.add_argument('drlhp_ckpt')
args = parser.parse_args()

seaquest_register()
breakout_register()
fetch_reach_register()
env = gym.make(args.env_id)  # type: TimeLimit
env._max_episode_seconds = None
env._max_episode_steps = None

if 'Fetch' in args.env_id:
    net = net_mlp
    network_args = {}
else:
    net = net_cnn
    network_args = {'batchnorm': False, 'dropout': 0.5}
reward_predictor = RewardPredictor(network=net,
                                   network_args=network_args,
                                   log_dir=tempfile.mkdtemp(),
                                   obs_shape=env.observation_space.shape,
                                   r_std=999,  # should be ignored
                                   name='test')
reward_predictor.load(args.drlhp_ckpt)

env = DrawRewards(env, reward_predictor)
env = RenderObs(env)
if 'Fetch' in args.env_id:
    env = FetchDiscreteActions(env)

play(env, zoom=1, fps=30)
