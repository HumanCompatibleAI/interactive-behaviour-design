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


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from drlhp.reward_predictor import RewardPredictor
from drlhp.reward_predictor_core_network import net_cnn, net_mlp
from wrappers.seaquest_reward import register as seaquest_register
from wrappers.breakout_reward import register as breakout_register
from wrappers.fetch_reach import register as fetch_reach_register
from wrappers.util_wrappers import FetchDiscreteActions


class DrawPredictedReward(Wrapper):
    def __init__(self, env, reward_predictor: RewardPredictor):
        super().__init__(env)
        self.reward_predictor = reward_predictor
        self.graph_values = deque(maxlen=100)
        self.last_obs = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_obs = np.array(obs)  # in case of LazyFrames
        return obs, reward, done, info

    def render(self, mode='human', **kwargs):
        assert mode == 'rgb_array'
        frame = self.env.render(mode='rgb_array')
        if self.last_obs is None:
            return frame
        r = self.reward_predictor.raw_rewards(np.array([self.last_obs]))[0][0]
        self.graph_values.append(r)
        frame[30, 5:-5, :] = 255
        frame[20, 5:-5, :] = 255
        frame[10, 5:-5, :] = 255
        frame[10:30, 5, :] = 255
        frame[10:30, -5, :] = 255
        for x, val in enumerate(self.graph_values):
            scale = np.max(np.abs(self.graph_values))
            y = int(val / scale * 10)
            frame[20 - y, 5 + x, :] = 255
        cv2.putText(frame,
                    "{:.3f}".format(r),
                    org=(20, 50),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=0.5,
                    color=[255] * frame.shape[-1],
                    thickness=1)
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
env = gym.make(args.env_id)

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

env = DrawPredictedReward(env, reward_predictor)
env = RenderObs(env)
if 'Fetch' in args.env_id:
    env = FetchDiscreteActions(env)

play(env, zoom=1, fps=30)
