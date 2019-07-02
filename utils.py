import code
import faulthandler
import glob
import gzip
import json
import lzma
import multiprocessing
import os
import pickle
import queue
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque, namedtuple, defaultdict
from functools import partial
from multiprocessing import Queue
from os import path as osp
from threading import Thread

import cv2
import easy_tf_log
import numpy as np
import psutil
import tensorflow as tf
from gym import Wrapper, Env
from gym.envs.atari import AtariEnv
from gym.envs.box2d import LunarLander
from gym.envs.mujoco import MujocoEnv
from gym.envs.robotics import FetchEnv
from gym.envs.robotics.robot_env import RobotEnv
from gym.spaces import Box, Discrete
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from tensorflow.python.client import device_lib

import global_variables
from baselines.ddpg.noise import ActionNoise
from wrappers.dummy_env import DummyEnv


def acquire_lock(env_lock, instigator):
    while True:
        acquire_success = env_lock.acquire(blocking=True, timeout=3)
        if acquire_success:
            break
        else:
            print(str(instigator) + " waiting for env lock...")


class TimerContext(object):
    def __init__(self, name, stdout=True):
        self.name = name
        self.stdout = stdout
        self.duration_s = None

    def __enter__(self):
        self.t_start = time.time()

    def __exit__(self, type, value, traceback):
        t_end = time.time()
        duration = t_end - self.t_start
        self.duration_s = duration
        if duration < 1e-3:
            units = "us"
            duration *= 1e6
        elif duration < 1:
            units = "ms"
            duration *= 1e3
        else:
            units = "s"
        if self.stdout:
            print("'{}' took {:.1f} {} ({})".format(self.name, duration, units, time.time()))


class LogMilliseconds:
    _values = defaultdict(list)

    def __init__(self, name, logger: easy_tf_log.Logger, log_every=1):
        self.name = name
        self.logger = logger
        self.log_every = log_every

    def __enter__(self):
        self.t_start = time.time()

    def __exit__(self, type, value, traceback):
        t_end = time.time()
        duration_seconds = t_end - self.t_start
        duration_ms = duration_seconds * 1000
        if self.log_every == 1:
            self.logger.logkv(self.name, duration_ms)
        else:
            LogMilliseconds._values[self.name].append(duration_ms)
            if len(LogMilliseconds._values[self.name]) == self.log_every:
                self.logger.log_list_stats(self.name, self._values[self.name])
                LogMilliseconds._values[self.name] = []


class RateMeasure:
    def __init__(self):
        self.prev_t = self.prev_value = None

    def reset(self, val):
        self.prev_value = val
        self.prev_t = time.time()

    def measure(self, val):
        val_change = val - self.prev_value
        cur_t = time.time()
        interval = cur_t - self.prev_t
        rate = val_change / interval

        self.prev_t = cur_t
        self.prev_value = val

        return rate


class Timer:
    """
    A simple timer class.
    * Set the timer duration with the `duration_seconds` argument to the constructor.
    * Start the timer by calling `reset()`.
    * Check whether the timer is done by calling `done()`.
    """

    def __init__(self, duration_seconds):
        self.duration_seconds = duration_seconds
        self.start_time = None

    def reset(self):
        self.start_time = time.time()

    def done(self):
        cur_time = time.time()
        if cur_time - self.start_time > self.duration_seconds:
            return True
        else:
            return False


def unwrap_to(wrapped_env: Wrapper, class_name: type, n_before=0):
    envs = deque(maxlen=(n_before + 1))
    env = wrapped_env
    envs.append(env)
    ex = Exception("Wrapper '{}' not found".format(class_name))
    while type(env) != class_name:
        if type(env) == AtariEnv:
            raise ex
        try:
            env = env.env
            envs.append(env)
        except:
            raise ex
    return envs[0]


def unwrap_to_instance(wrapped_env: Wrapper, class_name: type, n_before=0):
    envs = deque(maxlen=(n_before + 1))
    env = wrapped_env
    envs.append(env)
    ex = Exception("Wrapper '{}' not found".format(class_name))
    while not isinstance(env, class_name):
        if type(env) == AtariEnv:
            raise ex
        try:
            env = env.env
            envs.append(env)
        except:
            raise ex
    return envs[0]


def find_latest_checkpoint(ckpt_prefix):
    meta_paths = glob.glob(ckpt_prefix + '*.meta')
    if not meta_paths:
        raise Exception(f"Couldn't find checkpoint matching '{ckpt_prefix}*'")
    meta_path_timestample_tuples = []
    for p in meta_paths:
        try:
            meta_path_timestample_tuples.append((p, os.path.getmtime(p)))
        except FileNotFoundError:
            # One of the checkpoints got deleted since when we globbed; no big deal
            continue
    meta_path_timestample_tuples.sort(key=lambda tup: tup[1])
    ckpt_paths = [meta_path.replace('.meta', '') for meta_path, _ in meta_path_timestample_tuples]
    return ckpt_paths[-1]


class CompressedAttributes:
    """
    Base class with automagic attribute compression.

    Supports pickling.
    """

    def __init__(self, compression='lzma'):
        if compression not in ['lzma', 'gzip']:
            raise Exception(f"Unknown compression '{compression}'")
        self.compression = compression
        self.set_lib()
        self._data = dict()

    def set_lib(self):
        if self.compression == 'lzma':
            # preset=0 => fastest possible compression
            self.compress = partial(lzma.compress, preset=0)
            self.decompress = lzma.decompress
        elif self.compression == 'gzip':
            # compresslevel=1 => fastest possible compression (0 is no compression)
            self.compress = partial(gzip.compress, compresslevel=1)
            self.decompress = gzip.decompress

    def __getattr__(self, name):
        # __getattr__ is only called if the attribute isn't found the normal way.
        # So this won't be called for _data.
        try:
            return pickle.loads(self.decompress(self._data[name]))
        except KeyError:
            raise AttributeError

    def __setattr__(self, name, value):
        # __setattr__, on the other hand, is _always_ called.
        # So we need to be careful about _data.
        if name in ['_data', 'compression', 'compress', 'decompress']:
            object.__setattr__(self, name, value)
        else:
            self._data[name] = self.compress(pickle.dumps(value))

    def __getstate__(self):
        return (self._data, self.compression)

    def __setstate__(self, state):
        self._data = state[0]
        self.compression = state[1]
        self.set_lib()


def save_video(frames, vid_path):
    encoder = ImageEncoder(vid_path, frames[0].shape, frames_per_sec=30)
    for frame in frames:
        encoder.capture_frame(frame)
    encoder.close()


def concatenate_and_write_videos_to(input_vid_paths, output_vid_path):
    with tempfile.NamedTemporaryFile(mode='wt') as input_list_file:
        for v in input_vid_paths:
            input_list_file.write(f"file '{v}'\n")
        input_list_file.flush()
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                 '-i', input_list_file.name, '-c', 'copy', output_vid_path],
                check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: concatenating videos to {output_vid_path} failed:", e)


def get_env_state(env):
    if isinstance(env.unwrapped, AtariEnv):
        return env.unwrapped.clone_full_state()
    elif isinstance(env.unwrapped, MujocoEnv):
        return env.unwrapped.sim.get_state()
    elif isinstance(env.unwrapped, (FetchEnv, RobotEnv)):
        return (env.unwrapped.goal, env.unwrapped.sim.get_state())
    elif isinstance(env.unwrapped, LunarLander):
        # Handled by pickling
        return None
    elif isinstance(env.unwrapped, DummyEnv):
        return env.unwrapped.step_n
    else:
        raise Exception("Unsure how to save state for env type {}".format(type(env.unwrapped)))


def set_env_state(env, state):
    if isinstance(env.unwrapped, AtariEnv):
        env.unwrapped.restore_full_state(state)
    elif isinstance(env.unwrapped, MujocoEnv):
        env.unwrapped.sim.set_state(state)
    elif isinstance(env.unwrapped, (FetchEnv, RobotEnv)):
        env.unwrapped.goal = state[0]
        env.unwrapped.sim.set_state(state[1])
        env.unwrapped.sim.forward()
    elif isinstance(env.unwrapped, LunarLander):
        # Handled by pickling
        pass
    elif isinstance(env.unwrapped, DummyEnv):
        env.unwrapped.step_n = state
    else:
        raise Exception("Unsure how to restore state for env type {}".format(type(env.unwrapped)))


class EnvState(CompressedAttributes):
    def __init__(self, env: Env, obs: np.ndarray, done, step_n=None, birthtime=None):
        CompressedAttributes.__init__(self, 'gzip')
        self.obs = obs
        self.done = done
        self.state = get_env_state(env)
        self._env_pickle = pickle.dumps(env)
        self.step_n = step_n
        self.birthtime = birthtime

    @property
    def env(self):
        if global_variables.env_creation_lock is None:
            raise Exception("env_creation_lock not initialised")
        global_variables.env_creation_lock.acquire()
        env = pickle.loads(self._env_pickle)  # type: Env
        global_variables.env_creation_lock.release()
        set_env_state(env, self.state)
        return env


def get_noop_action(env):
    if isinstance(env.action_space, Discrete):
        try:
            action_meanings = env.unwrapped.get_action_meanings()
        except AttributeError:
            raise Exception(f"Couldn't determine no-op action for {env}")
        else:
            return action_meanings.index('NOOP')
    elif isinstance(env.action_space, Box):
        # Continuous action space; assume 0 on each dimension means "do nothing"
        return np.zeros(env.action_space.shape)
    else:
        raise Exception(f"Unsure how to determine no-op action for action space {env.action_space}")


def batch_iter(data, batch_size, shuffle=False):
    idxs = list(range(len(data)))
    if shuffle:
        np.random.shuffle(idxs)  # in-place

    start_idx = 0
    end_idx = 0
    while end_idx < len(data):
        end_idx = start_idx + batch_size
        if end_idx > len(data):
            end_idx = len(data)

        batch_idxs = idxs[start_idx:end_idx]
        batch = [data[i] for i in batch_idxs]
        yield batch

        start_idx += batch_size

    if len(data) != len(idxs):
        raise RuntimeError("list changed size during iteration")


def make_small_change(frames):
    frames = np.copy(frames)
    frames_flat = np.ravel(frames)
    idx = np.random.randint(low=0, high=len(frames_flat))
    r = np.random.randint(low=1, high=10)
    if frames_flat[idx] > 128:
        frames_flat[idx] -= r
    else:
        frames_flat[idx] += r
    return list(frames)


def save_args(args, save_dir, name='args.txt'):
    with open(osp.join(save_dir, name), 'w') as args_file:
        args_file.write(' '.join(sys.argv))
        args_file.write('\n')
        args_file.write(str(args))


class MemoryProfiler:
    STOP_CMD = 0

    def __init__(self, pid, log_path, include_children=False):
        self.pid = pid
        self.log_path = log_path
        self.cmd_queue = Queue()
        self.t = None
        self.include_children = include_children

    def start(self):
        self.t = Thread(target=self.profile)
        self.t.start()

    def stop(self):
        if self.t.is_alive():
            self.cmd_queue.put(self.STOP_CMD)
            self.t.join()
        self.t = None

    def profile(self):
        import memory_profiler
        f = open(self.log_path, 'w+')
        while True:
            # 1 sample
            try:
                memory_profiler.memory_usage(self.pid, stream=f, timeout=None, interval=0.1,
                                             include_children=self.include_children)
            except:
                # E.g. if the process has stopped existing since when we checked
                f.close()
                break

            f.flush()
            time.sleep(5)

            try:
                cmd = self.cmd_queue.get(timeout=0.1)
            except queue.Empty:
                pass
            else:
                if cmd == self.STOP_CMD:
                    f.close()
                    break

            if not psutil.pid_exists(self.pid):
                f.close()
                break


def sample_demonstration_batch(demonstration_rollouts, batch_size=32):
    if not demonstration_rollouts:
        raise NotEnoughDemonstrations("No demonstrations available")
    if len(demonstration_rollouts) < batch_size:
        sampled_rollouts = list(demonstration_rollouts.values())
    else:
        sampled_rollouts = random.sample(list(demonstration_rollouts.values()),
                                         batch_size)
    bc_obs = [obs
              for rollout in sampled_rollouts
              for obs in rollout.obses]
    bc_actions = [action
                  for rollout in sampled_rollouts
                  for action in rollout.actions]
    assert len(bc_obs) == len(bc_actions)
    return bc_obs, bc_actions


class NotEnoughDemonstrations(Exception):
    pass


def draw_dict_on_image(im, values_dict, mode='overlay'):
    line_height = 26
    alpha = 0.5

    im_shape = np.array(im).shape
    if len(im_shape) == 2:
        n_channels = 1
    elif len(im_shape) == 3:
        n_channels = im_shape[-1]
    else:
        raise Exception("Unsure how to draw on shape", im_shape)

    im_copy = np.copy(im)

    draw_height = len(values_dict) * line_height
    if mode == 'overlay':
        im_with_black_box = np.copy(im_copy)
        cv2.rectangle(im_with_black_box,
                      (0, 0),
                      (im_shape[1], draw_height),
                      thickness=cv2.FILLED,
                      color=[0] * n_channels)
        im_copy[:] = (1 - alpha) * np.array(im_copy) + alpha * im_with_black_box
    elif mode == 'concat':
        im_copy = np.vstack([np.zeros((draw_height,) + im_shape[1:]),
                             255 * np.ones((1,) + im_shape[1:]),
                             im_copy])
        # Ensure height is divisible by 2 for video encoding
        if im_copy.shape[2] % 2 != 0:
            im_copy = np.vstack([np.zeros((1,) + im_shape[1:]),
                                 im_copy])
    else:
        raise ValueError("Invalid mode {}".format(mode))

    for n, (key, value) in enumerate(values_dict.items()):
        if isinstance(value, float):
            value = np.float32(value)

        if isinstance(value, (np.ndarray, np.float32)):
            value_str = np.array2string(value, precision=3, sign=' ', floatmode='fixed', suppress_small=True)
        else:
            value_str = str(value)
        y = n * line_height
        cv2.putText(im_copy,
                    str(key) + ': ' + value_str,
                    org=(5, 18 + y),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.0,
                    color=(255, 255, 255),
                    thickness=1)

    if isinstance(im, np.ndarray):
        im_copy = im_copy.astype(im.dtype)

    return im_copy


def get_git_rev():
    # Used for code uploaded to AWS
    if os.path.exists('gitrev'):
        with open('gitrev', 'r') as f:
            gr = f.read().strip()
        return gr

    try:
        cmd = 'git rev-parse --short HEAD'
        git_rev = subprocess.check_output(cmd.split(' '), stderr=subprocess.PIPE).decode().rstrip()
        return git_rev
    except subprocess.CalledProcessError:
        return 'unkrev'


class RunningProportion:
    def __init__(self):
        self.n = 0
        self.v = 0

    def update(self, v):
        self.v = ((self.v * self.n) + v) / (self.n + 1)
        self.n += 1


def load_cpu_config(log_dir, name):
    with open(os.path.join(log_dir, 'cpus.json'), 'r') as f:
        cpus = json.load(f)[name]
    # This specifically needs to be done before starting the TensorFlow execution engine so that child threads
    # have their affinity also set.
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, cpus)
    else:
        assert sys.platform == 'darwin'
    # The first session created in a process seems to set the options for the execution engine used for all subsequent
    # session in that process. So this should be called before anything else.
    config = tf.ConfigProto(inter_op_parallelism_threads=len(cpus), intra_op_parallelism_threads=len(cpus))
    config.gpu_options.allow_growth = True
    dummy_sess = tf.Session(config=config)
    dummy_sess.close()


def save_cpu_config(log_dir, main_cpus, rollouter_cpus, drlhp_training_cpus):
    d = {'main': main_cpus,
         'rollouters': rollouter_cpus,
         'drlhp_training': drlhp_training_cpus}
    with open(os.path.join(log_dir, 'cpus.json'), 'w') as f:
        json.dump(d, f)


def configure_cpus(log_dir, n_rollouter_cpus=6, n_drlhp_training_cpus=4):
    if hasattr(os, 'sched_getaffinity'):
        available_cpus = list(os.sched_getaffinity(0))
    else:
        assert sys.platform == 'darwin'
        available_cpus = list(range(multiprocessing.cpu_count()))
    rollouter_cpus = [available_cpus.pop() for _ in range(n_rollouter_cpus)]
    n_drlhp_training_cpus = n_drlhp_training_cpus
    drlhp_training_cpus = [available_cpus.pop() for _ in range(n_drlhp_training_cpus)]
    main_cpus = available_cpus
    save_cpu_config(log_dir, main_cpus, rollouter_cpus, drlhp_training_cpus)


# https://stackoverflow.com/a/133384/7832197

def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d = {'_frame': frame}  # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)

    faulthandler.dump_traceback(all_threads=True)
    i = code.InteractiveConsole(d)
    i.interact('Entering shell')


def register_debug_handler():
    signal.signal(signal.SIGUSR1, debug)  # Register handler


class LimitedRunningStat:
    def __init__(self, shape, len=10000):
        self.values = np.zeros((len,) + shape)
        self.n_values = 0
        self.i = 0

    def push(self, x):
        self.values[self.i] = x
        self.i = (self.i + 1) % len(self.values)
        if self.n_values < len(self.values):
            self.n_values += 1

    @property
    def n(self):
        return self.n_values

    @property
    def mean(self):
        return np.mean(self.values[:self.n_values], axis=0)

    @property
    def var(self):
        return np.var(self.values[:self.n_values], axis=0)

    @property
    def std(self):
        return np.std(self.values[:self.n_values], axis=0)


class NoGPUsError(Exception):
    pass


class NoFreeGPUsError(Exception):
    pass


def find_least_busy_gpu():
    try:
        output = subprocess.check_output(['nvidia-smi']).decode()
        lines = output.rstrip().split('\n')
    except FileNotFoundError:
        raise NoGPUsError()
    mem_usage_frac_by_gpu = []
    next_line_is_stats = False
    for line in lines:
        if next_line_is_stats:
            mem_stats = line.split('|')[2]
            m = re.search(r'(.*)MiB / (.*)MiB', mem_stats)
            used = float(m.group(1))
            total = float(m.group(2))
            print(used, total)
            mem_usage_frac = used / total
            mem_usage_frac_by_gpu.append(mem_usage_frac)
        if any(g in line for g in ['TITAN', 'GeForce']):
            next_line_is_stats = True
        else:
            next_line_is_stats = False
    if not any([frac < 0.8 for frac in mem_usage_frac_by_gpu]):
        raise NoFreeGPUsError()
    least_busy_gpu_n = np.argmin(mem_usage_frac_by_gpu)
    return least_busy_gpu_n


def get_available_gpu_ns_proc(results_queue):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Prevent list_local_devices taking up all GPU memory
    gpu_ns = [x.name.split(':')[2]
              for x in device_lib.list_local_devices(session_config=config)
              if x.device_type == 'GPU']
    results_queue.put(gpu_ns)


def get_available_gpu_ns():
    """
    Get available GPUs.

    We have to use a separate process because otherwise we might start the TensorFlow execution engine
    for this process - which is a problem if we haven't configured CPU binding yet.
    """
    ctx = multiprocessing.get_context('spawn')
    results_queue = ctx.Queue()
    ctx.Process(target=get_available_gpu_ns_proc, args=(results_queue,)).start()
    return results_queue.get()


def read_events_file(events_filename):
    events = {}
    try:
        for event in tf.train.summary_iterator(events_filename):
            for value in event.summary.value:
                if value.tag not in events:
                    events[value.tag] = []
                events[value.tag].append((event.wall_time, value.simple_value))
    except Exception as e:
        print(f"While reading '{events_filename}':", e)
    return events


# From Baselines DDPG
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, seed, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.np_random = None
        self.reseed(seed)
        self.reset()

    def __call__(self):
        rand_normal = self.np_random.normal(size=self.mu.shape)
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * rand_normal
        self.x_prev = x
        return x

    def reseed(self, seed):
        self.np_random = np.random.RandomState(seed=seed)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


ObsRewardTuple = namedtuple('ObsRewardTuple', 'obs reward')


def split_preserving_seps(s, seps=('_', '-')):
    """
    'foo-bar_1' => ['foo', '-', 'bar', '_', '1']
    """
    arr = []
    cur = ""
    for c in s:
        if c in seps:
            if cur:
                arr.append(cur)
            arr.append(c)
            cur = ""
        else:
            cur += c
    arr.append(cur)
    return arr