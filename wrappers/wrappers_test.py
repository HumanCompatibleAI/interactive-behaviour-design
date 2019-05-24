#!/usr/bin/env python

import multiprocessing
import os
import sys
import threading
import unittest

import numpy as np
from gym import Env

import global_constants
from a2c.common.vec_env.subproc_vec_env import SubprocVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import global_variables
from wrappers.dummy_env import DummyEnv
from wrappers.util_wrappers import SaveMidStateWrapper, RepeatActions, VecSaveSegments
from wrappers.state_boundary_wrapper import StateBoundaryWrapper


class TestSaveMidState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Needed to be able to restore env from state
        global_variables.env_creation_lock = threading.Lock()

    def test_distribution(self):
        """
        Confirm that distribution of steps from which states are saved is approximately uniform.
        """
        env = DummyEnv(max_steps=100)
        env = StateBoundaryWrapper(env)
        q = multiprocessing.Queue()
        np.random.seed(0)
        env = SaveMidStateWrapper(env, q, verbose=False)
        env.reset()
        states = []
        for _ in range(1000):
            done = False
            while not done:
                obs, _, done, _ = env.step(action=None)
            env.reset()
            states.append(q.get())

        state_steps = [state.step_n for state in states]
        # Assuming states saved at intervals of 10 steps
        assert env.SAVE_EVERY_NTH_STEP == 10
        counts, bins = np.histogram(state_steps, bins=np.arange(5, 115, 10))
        np.testing.assert_allclose(counts, 10 * [100], atol=20)

    def test_short_episodes(self):
        """
        Check that a state gets saved even for short episodes
        """
        q = multiprocessing.Queue()
        states = []
        for n_extra_steps in range(10):
            env = DummyEnv(max_steps=SaveMidStateWrapper.SAVE_EVERY_NTH_STEP - 5 + n_extra_steps)
            env = StateBoundaryWrapper(env)
            env = SaveMidStateWrapper(env, q, verbose=False)
            env.reset()
            done = False
            while not done:
                obs, _, done, _ = env.step(action=None)
            env.reset()
            states.append(q.get())
        self.assertTrue(all([state.env.step_n > 0 for state in states]))


class TestRepeatActions(unittest.TestCase):
    def test(self):
        class DummyEnv(Env):
            def __init__(self):
                Env.__init__(self)
                self.step_n = None
                self.actions = None

            def _obs(self):
                return self.step_n

            def reset(self):
                self.step_n = 0
                self.actions = []
                return self._obs()

            def step(self, action):
                self.actions.append(action)
                self.step_n += 1
                reward = action
                done = False
                info = None
                return self._obs(), reward, done, info

        env = DummyEnv()
        env = RepeatActions(env, repeat_n=1)
        env.reset()
        rewards = []
        for n in range(3):
            obs, reward, done, info = env.step(n)
            rewards.append(reward)
        assert env.unwrapped.actions == [0, 1, 2], env.unwrapped.actions
        assert rewards == [0, 1, 2], rewards

        env = DummyEnv()
        env = RepeatActions(env, repeat_n=2)
        env.reset()
        rewards = []
        for n in range(3):
            obs, reward, done, info = env.step(n)
            rewards.append(reward)
        assert env.unwrapped.actions == [0, 0, 1, 1, 2, 2], env.unwrapped.actions
        assert rewards == [0, 2, 4], rewards


class TestVecSaveSegments(unittest.TestCase):
    def test(self):
        segments_queue = multiprocessing.Queue()
        n_envs = 3
        venv = SubprocVecEnv([lambda n=n: StateBoundaryWrapper(DummyEnv(global_constants.FRAMES_PER_SEGMENT - 1 + n,
                                                                        step_offset=(n * 100)))
                              for n in range(n_envs)])
        venv = VecSaveSegments(venv, segments_queue)

        venv.reset()
        for _ in range(3 * global_constants.FRAMES_PER_SEGMENT):
            venv.step([0] * venv.num_envs)

        obses, rewards, frames = segments_queue.get()
        # The first segment we get should be from the environment that reset after 29 steps
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 28]
        np.testing.assert_array_equal(obses, expected)

        obses, rewards, frames = segments_queue.get()
        # The next should be from the environment that resets after exactly 30 steps
        expected = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                    113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                    126, 127, 128, 129]
        np.testing.assert_array_equal(obses, expected)

        # Then from the environment which resets after 31 steps
        obses, rewards, frames = segments_queue.get()
        expected = [200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
                    213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225,
                    226, 227, 228, 229]
        np.testing.assert_array_equal(obses, expected)

        # Shortly after, the 31-step environment should get a 'done', resulting in a single-step segment
        # padded to 30 steps
        obses, rewards, frames = segments_queue.get()
        expected = [230] * 30
        np.testing.assert_array_equal(obses, expected)

        # Then we start again with a segment from the 29-step environment
        obses, rewards, frames = segments_queue.get()
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 28]
        np.testing.assert_array_equal(obses, expected)


if __name__ == '__main__':
    unittest.main()
