import unittest

import numpy as np
from gym import Env

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv as BaselinesSubprocVecEnv
from subproc_vec_env_custom import CustomSubprocVecEnv as CustomSubprocVecEnv


class DummyEnv(Env):
    def __init__(self, max_steps=10):
        super().__init__()
        self.step_n = None
        self.max_steps = max_steps

    def reset(self):
        self.step_n = 0
        return self.step_n

    def step(self, action):
        self.step_n += 1
        obs = self.step_n
        reward = None
        done = (self.step_n >= self.max_steps)
        info = None
        return obs, reward, done, info


def manual_check():
    e = DummyEnv()
    for _ in range(2):
        print(e.reset())
        done = False
        while not done:
            obs, reward, done, info = e.step(None)
            print(obs, reward, done, info)

    print()

    e = BaselinesSubprocVecEnv([lambda: DummyEnv()])
    for _ in range(2):
        print(e.reset())
        dones = [False]
        while not dones[0]:
            obses, rewards, dones, infos = e.step([None])
            print(obses, rewards, dones, infos)
    e.close()

    print()

    e = CustomSubprocVecEnv([lambda: DummyEnv()])
    for _ in range(2):
        print(e.reset())
        dones = [False]
        while not dones[0]:
            obses, rewards, dones, infos = e.step([None])
            print(obses, rewards, dones, infos)
    e.close()


class Test(unittest.TestCase):
    def test_sync(self):
        e = CustomSubprocVecEnv([lambda: DummyEnv(max_steps=10)] * 2)
        obses_list = []
        for _ in range(2):
            obses_list.append(e.reset())
            dones = [False] * 2
            while not (any(dones)):
                obses, rewards, dones, infos = e.step([None] * 2)
                obses_list.append(obses)
        obses_list = np.array(obses_list)
        expected = np.tile(np.arange(11), (2, 2)).transpose()
        np.testing.assert_array_equal(obses_list, expected)

    def test_async(self):
        e = CustomSubprocVecEnv([lambda: DummyEnv(max_steps=2), lambda: DummyEnv(max_steps=3)])
        obses1 = []
        obses2 = []
        o1, o2 = e.reset()
        obses1.append(o1)
        obses2.append(o2)

        for _ in range(9):
            (o1, o2), rewards, dones, infos = e.step([None] * 2)
            obses1.append(o1)
            obses2.append(o2)
            if dones[0]:
                obses1.append(e.reset_one_env(0))
            if dones[1]:
                obses2.append(e.reset_one_env(1))

        self.assertEqual(obses1, [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
        self.assertEqual(obses2, [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0])


if __name__ == '__main__':
    unittest.main()
