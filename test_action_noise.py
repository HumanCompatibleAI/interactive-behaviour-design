import argparse
import multiprocessing
import queue
import random
import time
from collections import Counter, defaultdict
from multiprocessing import Process, Queue

import gym
import gym.spaces
import numpy as np
from gym.envs.classic_control import rendering
from gym.wrappers.monitoring.video_recorder import VideoRecorder, ImageEncoder

import global_variables
from utils import EnvState
from wrappers.seaquest_reward import register as seaquest_register
from wrappers.breakout_reward import register as breakout_register, BreakoutRewardWrapper
from wrappers.wrappers_debug import NumberFrames

seaquest_register()
breakout_register()

n_envs = 3
barrier = multiprocessing.Barrier(n_envs)

parser = argparse.ArgumentParser()
parser.add_argument('action_selection', choices=['sample', 'correlated'])
args = parser.parse_args()


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


class Oracle():
    def __init__(self, env):
        self.last_action = None
        self.env = env

    def get_action(self):
        if self.last_action is not None and random.random() < 0.8:
            action = self.last_action
        else:
            action = self.env.action_space.sample()
        self.last_action = action
        return action


def f(env_n, frame_queue, state_queue, best_state_queue):
    global_variables.env_creation_lock = multiprocessing.Lock()
    env = gym.make('BreakoutNoFrameskip-v4')
    env = BreakoutRewardWrapper(env)
    env = NumberFrames(env)
    gym.spaces.seed(env_n)
    env.reset()
    rewards = []
    oracle = Oracle(env)
    actions = []
    while True:
        # frame_queue.put((env_n, env.render(mode='rgb_array')))
        if args.action_selection == 'sample':
            action = env.action_space.sample()
        elif args.action_selection == 'correlated':
            action = oracle.get_action()
        else:
            raise Exception()
        actions.append(action)
        obs, reward, done, info = env.step(action)
        # time.sleep(1/20)
        rewards.append(reward)
        if done or len(rewards) == 25:
            if done:
                env.reset()
            barrier.wait()
            state_queue.put((env_n, sum(rewards), EnvState(env, None, None)))
            env = best_state_queue.get().env
            rewards = []
            barrier.wait()


def render_loop(frame_queue, n_envs):
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


def state_loop(state_queue, best_state_queue, n_envs):
    states = {}
    d = defaultdict(lambda: 0)
    r = 0
    n = 0
    while True:
        env_n, reward, state = state_queue.get()
        states[env_n] = (reward, state)
        if len(states) == n_envs:
            # for k, v in states.items():
            #     print(k, v)
            d[len(set([t[0] for t in states.values()]))] += 1
            best_reward_state = sorted(states.values(), key=lambda tup: tup[0])[-1]
            r += best_reward_state[0]
            print(n, r)
            for _ in range(n_envs):
                best_state_queue.put(best_reward_state[1])
            states = {}
            # print()
            n += 1


frame_queue, state_queue, best_state_queue = Queue(maxsize=n_envs * 3), Queue(), Queue()
for n in range(n_envs):
    Process(target=f, args=(n, frame_queue, state_queue, best_state_queue)).start()
Process(target=state_loop, args=(state_queue, best_state_queue, n_envs)).start()
# render_loop(frame_queue, n_envs)
