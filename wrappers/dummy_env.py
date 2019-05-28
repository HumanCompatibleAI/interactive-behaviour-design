import cv2
from gym import Env
from gym.envs.registration import EnvSpec
import numpy as np


class DummyEnv(Env):
    def __init__(self, max_steps, step_offset=0):
        self.step_n = None
        self.max_steps = max_steps
        self.step_offset = step_offset
        self.spec = EnvSpec('DummyEnv-v0')

    def step(self, action):
        obs = self._obs()
        reward = 0
        info = None
        if self.step_n < self.step_offset + self.max_steps:
            done = False
        else:
            done = True
        self.step_n += 1
        return obs, reward, done, info

    def reset(self):
        self.step_n = self.step_offset
        return self._obs()

    def _obs(self):
        return self.step_n

    def render(self, mode='human'):
        assert mode == 'rgb_array'
        im = np.zeros((100, 100))
        cv2.putText(im,
                    str(self.step_n),
                    org=(0, 0),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2.0,
                    color=[255],
                    thickness=2)
        return im
