#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import pickle
import time

import gym
from gym.utils import atomic_write
from gym.wrappers import Monitor

from wrappers.lunar_lander_reward import LunarLanderStatsWrapper
from wrappers.util_wrappers import SaveEpisodeStats


class Demonstration:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.frames = []

    def append(self, obs, act, frame):
        self.observations.append(obs)
        self.actions.append(act)
        self.frames.append(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()

    env = gym.make('LunarLander-v2')
    env = Monitor(env, args.log_dir, video_callable=lambda n: True)
    env = LunarLanderStatsWrapper(env)
    env = SaveEpisodeStats(env, stdout=True, log_dir=args.log_dir)

    ACTIONS = env.action_space.n
    human_agent_action = 0
    human_wants_pause = False

    def key_press(key, mod):
        nonlocal human_agent_action, human_wants_pause
        if key == 32:
            human_wants_pause = not human_wants_pause
            return
        a = int(key - ord('0'))
        if a <= 0 or a >= ACTIONS: return
        human_agent_action = a

    def key_release(key, mod):
        nonlocal human_agent_action
        a = int(key - ord('0'))
        if a <= 0 or a >= ACTIONS: return
        if human_agent_action == a:
            human_agent_action = 0

    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    demonstrations = []

    while True:
        obs, done = env.reset(), False
        demonstration = Demonstration()
        while not done:
            t1 = time.time()
            demonstration.append(obs, human_agent_action, None)
            obs, r, done, info = env.step(human_agent_action)

            interval = time.time() - t1
            if interval < 1 / 20:
                time.sleep(1 / 20 - interval)

            t2 = time.time()
            interval = (t2 - t1)
            print("{:.1f} frames/second".format(1 / interval))

            while human_wants_pause:
                env.render()
                time.sleep(0.1)
        demonstrations.append(demonstration)
        with atomic_write.atomic_write(os.path.join(args.log_dir, 'demonstrations.pkl'),
                                       binary=True, fsync=True) as f:
            pickle.dump(demonstrations, f)


if __name__ == '__main__':
    main()
