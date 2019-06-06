import argparse
import multiprocessing
import queue
import random
from collections import defaultdict
from multiprocessing import Process, Queue
from time import sleep

import gym
import gym.spaces
import numpy as np
import pysnooper
from gym import Wrapper
from gym.envs.classic_control import rendering
from gym.spaces import Box
from gym.wrappers.monitoring.video_recorder import VideoRecorder, ImageEncoder

import global_variables
from baselines.common.distributions import OrnsteinUhlenbeckActionNoise
from utils import EnvState
from wrappers.breakout_reward import register as breakout_register
from wrappers.seaquest_reward import register as seaquest_register
from wrappers.fetch_pick_and_place_register import register as fetch_register

seaquest_register()
breakout_register()
fetch_register()


class VideoRecorder(object):
    def __init__(self, path):
        self.encoder = None  # lazily start the process
        self.path = path

    def write_frame(self, frame):
        self._encode_image_frame(frame)

    def _encode_image_frame(self, frame):
        if not self.encoder:
            self.encoder = ImageEncoder(self.path, frame.shape, 60)
        self.encoder.capture_frame(frame)


def compress(l):
    l2 = []
    prev = l[0]
    count = 1
    for x in l[1:]:
        if x != prev:
            l2.append((prev, count))
            prev = x
            count = 1
        else:
            count += 1
    l2.append((prev, count))
    return l2


class Oracle():
    def __init__(self, env, correlation):
        self.last_action = None
        self.env = env
        self.correlation = correlation

    def get_action(self):
        if self.last_action is not None and random.random() < self.correlation:
            action = self.last_action
        else:
            action = self.env.action_space.sample()
        self.last_action = action
        return action


class ObsRender(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84, 84))
        self.last_obs = np.zeros((84, 84, 3))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_obs = np.tile(obs[:, :, 0][:, :, np.newaxis], [1, 1, 3])
        return obs, reward, done, info

    def render(self, mode='human', **kwargs):
        return self.last_obs


# @pysnooper.snoop()
def env_worker(env_n, frame_queue, state_queue, best_state_queue, env_done_barrier, seg_len, action_selection, render, action_correlation):
    global_variables.env_creation_lock = multiprocessing.Lock()
    env = gym.make(
        'FetchPickAndPlace-Repeat1-ContGripper-WithGripObs-InfInitialBlockPos-FixedGoal-Delta-GripperBonuses-v0')
    gym.spaces.seed(env_n)
    env.reset()
    rewards = []
    oracle = Oracle(env, action_correlation)
    mu = np.zeros(env.action_space.shape)
    sigma = 0.2 * np.ones(env.action_space.shape)
    ou = OrnsteinUhlenbeckActionNoise(mu=mu, sigma=sigma)
    actions = []
    while True:
        if render:
            frame = env.render(mode='rgb_array')
            frame_queue.put((env_n, frame))
        if action_selection == 'sample':
            action = env.action_space.sample()
        elif action_selection == 'correlated':
            action = oracle.get_action()
        elif action_selection == 'ou':
            action = ou()
            action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
        else:
            raise Exception()
        actions.append(action)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if done or len(rewards) == seg_len:
            ou = OrnsteinUhlenbeckActionNoise(mu=mu, sigma=sigma)
            if done:
                env.reset()
            env_done_barrier.wait()
            # noinspection PyTypeChecker
            state_queue.put((env_n, sum(rewards), EnvState(env, None, None)))
            env = best_state_queue.get().env
            rewards = []
            env_done_barrier.wait()


def frame_render_loop(frame_queue, n_envs):
    viewer = rendering.SimpleImageViewer()
    video_n = 0
    recorder = VideoRecorder('out-0.mp4')
    all_frames = None
    n = 0
    while True:
        try:
            env_n, frame = frame_queue.get()
        except queue.Empty:
            continue
        h = frame.shape[0]
        w = frame.shape[1]
        d = frame.shape[2]
        if all_frames is None:
            all_frames = np.zeros((h, n_envs * w, d), dtype=np.uint8)
        all_frames[:, env_n * w:(env_n + 1) * w] = np.copy(frame)
        viewer.imshow(all_frames)
        recorder.write_frame(all_frames)

        n += 1
        if n % (30 * 10) == 0:
            recorder.encoder.close()
            video_n += 1
            recorder = VideoRecorder(f'out-{video_n}.mp4')


def state_choose_loop(state_queue, best_state_queue, n_envs):
    reward_state_tuples = {}
    n = 0
    while True:
        env_n, reward, state = state_queue.get()
        reward_state_tuples[env_n] = (reward, state)
        if len(reward_state_tuples) == n_envs:
            for env_n in sorted(reward_state_tuples.keys()):
                print(env_n, reward_state_tuples[env_n])
            best_reward_state_tuple = sorted(reward_state_tuples.values(), key=lambda tup: tup[0])[-1]
            for _ in range(n_envs):
                best_state_queue.put(best_reward_state_tuple[1])
            reward_state_tuples = {}
            n += 1
            print(f"Completed {n} segments")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action_selection', choices=['sample', 'correlated', 'ou'])
    parser.add_argument('--correlation', type=float, default=0.99)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seg_len', type=int, default=60)
    parser.add_argument('--n_envs', type=int, default=3)
    args = parser.parse_args()

    ctx = multiprocessing.get_context('spawn')
    env_done_barrier = ctx.Barrier(args.n_envs)
    frame_queue, state_queue, best_state_queue = ctx.Queue(maxsize=args.n_envs), ctx.Queue(), ctx.Queue()
    for n in range(args.n_envs):
        worker_args = (n, frame_queue, state_queue, best_state_queue, env_done_barrier, args.seg_len, args.action_selection,
                   args.render, args.correlation)
        ctx.Process(target=env_worker, args=worker_args, daemon=True).start()
    ctx.Process(target=state_choose_loop, args=(state_queue, best_state_queue, args.n_envs), daemon=True).start()
    frame_render_loop(frame_queue, args.n_envs)


if __name__ == '__main__':
    main()
