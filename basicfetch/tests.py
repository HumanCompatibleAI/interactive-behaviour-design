"""
Manual tests. Run with e.g.:
    python tests.py Tests.initial_positions
"""

import unittest

import gym
import numpy as np
from pylab import *

from basicfetch import register

register()

np.set_printoptions(precision=3, floatmode='fixed', linewidth=999, suppress=True, sign=' ')


class Tests(unittest.TestCase):
    def plot_initial_positions(self):
        env = gym.make('FetchBasicDelta-v0')
        initial_positions = []
        for n in range(50):
            print(n)
            env.reset()
            gripper_position = env.unwrapped.sim.data.get_site_xpos('grip')
            initial_positions.append(np.copy(gripper_position))
        initial_positions = np.array(initial_positions)
        print(initial_positions)
        subplot(1, 3, 1)
        hist(initial_positions[:, 0])
        title('x')
        subplot(1, 3, 2)
        hist(initial_positions[:, 1])
        title('y')
        subplot(1, 3, 3)
        hist(initial_positions[:, 2])
        title('z')
        show()

    def show_initial_positions(self):
        env = gym.make('FetchBasicDelta-v0')
        while True:
            env.reset()
            done = False
            while not done:
                _, _, done, _ = env.step(np.zeros(env.action_space.shape))
                env.render()

    def delta_env(self):
        env = gym.make('FetchBasicDelta-v0').unwrapped  # unwrap past TimeLimit
        for random_initial_pos in [False, True]:
            env.unwrapped.random_initial_gripper_position = random_initial_pos
            for action_n in range(env.action_space.shape[0]):
                action = np.zeros(env.action_space.shape)
                for action_val in [-1, 0, +1]:
                    env.reset()
                    print(f"Action {action_n} = {action_val}")
                    action[action_n] = action_val
                    for _ in range(100):
                        env.step(action)
                        env.render()
                        ctrl = env.unwrapped.sim.data.ctrl
                        print(env.unwrapped.actuator_ctrl_to_normalized_action(ctrl))
                    input()


if __name__ == '__main__':
    unittest.main()
