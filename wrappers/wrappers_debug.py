import os
from collections import deque

import cv2
import gym.spaces as spaces
import numpy as np
from gym.core import ObservationWrapper, Wrapper
from gym.wrappers.monitoring.video_recorder import ImageEncoder

from baselines.common.vec_env import VecEnvWrapper
from utils import draw_dict_on_image, unwrap_to_instance
from wrappers.util_wrappers import CollectEpisodeStats

"""
Wrappers for gym environments to help with debugging.
"""


class NumberFrames(Wrapper):
    """
    Draw number of frames since reset.
    """

    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.frames_since_reset = None

    def reset(self):
        self.frames_since_reset = 0
        return self.observation(self.env.reset())

    def step(self, action):
        self.frames_since_reset += 1
        o, r, d, i = self.env.step(action)
        o = self.observation(o)
        return o, r, d, i

    def observation(self, obs):
        obs = np.array(obs)  # in case of LazyFrames
        cv2.putText(obs,
                    str(self.frames_since_reset),
                    org=(0, 70),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2.0,
                    color=[255] * obs.shape[-1],
                    thickness=2)
        return obs

    def render(self, mode='human', **kwargs):
        obs = self.env.render(mode='rgb_array', **kwargs)
        obs = self.observation(obs)
        return obs


class EarlyReset(Wrapper):
    """
    Reset the environment after 100 steps.
    """

    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.n_steps = None

    def reset(self):
        self.n_steps = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.n_steps += 1
        if self.n_steps >= 100:
            done = True
        return obs, reward, done, info


class ConcatFrameStack(ObservationWrapper):
    """
    Concatenate a stack horizontally into one long frame.
    """

    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        # Important so that gym's play.py picks up the right resolution
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(84, 4 * 84),
                                            dtype=np.uint8)

    def observation(self, obs):
        assert obs.shape[0] == 4
        return np.hstack(obs)


class DrawActions(Wrapper):
    def __init__(self, env):
        self.last_action = None
        Wrapper.__init__(self, env)

    def reset(self):
        self.last_action = None
        return self.env.reset()

    def step(self, action):
        self.last_action = action
        return self.env.step(action)

    def render(self, mode='human', **kwargs):
        if mode == 'rgb_array':
            im = self.env.render('rgb_array')
            im = draw_dict_on_image(im, {'actions': self.last_action},
                                    mode='concat')
            return im
        else:
            return self.env.render(mode)


class DrawRewards(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_reward = None
        self.ret = None

    def reset(self):
        self.ret = 0
        self.last_reward = None
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_reward = reward
        self.ret += reward
        return obs, reward, done, info

    def render(self, mode='human', **kwargs):
        if mode == 'rgb_array':
            im = self.env.render('rgb_array')
            im = draw_dict_on_image(im, {'reward': self.last_reward,
                                         'return': self.ret},
                                    mode='concat')
            return im
        else:
            return self.env.render(mode)


class RewardGrapher:
    def __init__(self, scale=None):
        self.scale = scale
        self.values = None
        self.reset()

    def reset(self):
        self.values = deque(maxlen=100)

    def draw(self, frame):
        if not self.values:
            return

        y = 20
        height = 100

        frame[y, 5:-5, :] = 255
        frame[y + height, 5:-5, :] = 255
        frame[y + height // 2, 5:-5, :] = 255
        frame[y:y + height, 5, :] = 255
        frame[y:y + height, -5, :] = 255

        if self.scale is None:
            scale = np.max(np.abs(self.values))
            if scale == 0:
                scale = 1
        else:
            scale = self.scale

        for x, val in enumerate(self.values):
            val_y = int((val / scale) * (height / 2))
            frame[y + height // 2 - val_y, 5 + x, :] = 255

        # For some reason putText can't draw directly on the original frame...?
        frame_copy = np.copy(frame)
        cv2.putText(frame_copy,
                    "{:.3f}".format(self.values[-1]),
                    org=(5, 15),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1,
                    color=[255, 255, 255],
                    thickness=1)
        frame[:] = frame_copy[:]


class GraphRewards(Wrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.reward_grapher = RewardGrapher(scale)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.reward_grapher.values.append(reward)
        return obs, reward, done, info

    def reset(self):
        self.reward_grapher.reset()
        return self.env.reset()

    def render(self, mode='human', **kwargs):
        assert mode == 'rgb_array'
        im = self.env.render(mode='rgb_array')
        self.reward_grapher.draw(im)
        return im


class VecRecordVideosWithPredictedRewardGraphs(VecEnvWrapper):
    def __init__(self, venv, video_dir, reward_predictor):
        self.encoder = None
        self.dir = video_dir
        os.makedirs(self.dir)
        self.episode_n = 0
        self.reward_grapher = RewardGrapher()
        self.reward_predictor = reward_predictor
        super().__init__(venv)

    def step_wait(self):
        obses, rewards, dones, infos = self.venv.step_wait()
        frames = self.venv.get_images()
        predicted_rewards = self.reward_predictor.raw_rewards(obses)[0]
        self.reward_grapher.values.append(predicted_rewards[0])
        frame = self.reward_grapher.draw(frames[0])

        if not self.encoder:
            video_fname = os.path.join(self.dir, f'{self.episode_n}.mp4')
            self.encoder = ImageEncoder(video_fname,
                                        frame.shape,
                                        frames_per_sec=30)
        self.encoder.capture_frame(frame)
        if dones[0]:
            self.encoder.close()
            self.encoder = None
        return obses, rewards, dones, infos

    def reset(self):
        self.episode_n += 1
        self.reward_grapher.reset()
        return self.venv.reset()

    def reset_one_env(self, env_n):
        if env_n == 0:
            self.episode_n += 1
        self.venv.reset_one_env(env_n)


class DrawObses(Wrapper):
    def __init__(self, env, decode_fn=None):
        self.last_obs = None
        self.decode_fn = decode_fn
        Wrapper.__init__(self, env)

    def reset(self):
        self.last_obs = self.env.reset()
        return self.last_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs
        return obs, reward, done, info

    def render(self, mode='human', **kwargs):
        if mode == 'rgb_array' and self.last_obs is not None:
            im = self.env.render('rgb_array')
            if self.decode_fn:
                d = self.decode_fn(self.last_obs)
            else:
                d = {'obs': self.last_obs}
            im = draw_dict_on_image(im, d, mode='concat')
            return im
        else:
            return self.env.render(mode, **kwargs)


class DrawStats(Wrapper):
    def reset(self):
        return self.env.reset()

    def render(self, mode='human', **kwargs):
        if mode != 'rgb_array':
            return self.env.render(mode, **kwargs)

        im = self.env.render('rgb_array')
        stats_wrapper = unwrap_to_instance(self.env, CollectEpisodeStats)
        im = draw_dict_on_image(im, stats_wrapper.stats, mode='concat')
        return im


class DrawEnvN(Wrapper):
    def __init__(self, env, n):
        super().__init__(env)
        self.n = n

    def reset(self, **kwargs):
        return self.env.reset()

    def render(self, mode='human', **kwargs):
        im = self.env.render(mode='rgb_array')
        im = draw_dict_on_image(im, {'env_n': self.n}, mode='concat')
        return im
