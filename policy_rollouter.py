import json
import multiprocessing
import os
import pickle
import queue
import sys
import time
from typing import Dict

import gym
import numpy as np
from cloudpickle import cloudpickle
from gym.spaces import prng
from gym.utils import atomic_write
from gym.wrappers import TimeLimit
from tensorflow.python.framework.errors_impl import NotFoundError

import global_variables
from global_constants import ROLLOUT_FPS
from global_variables import RolloutMode, RolloutRandomness
from rollouts import CompressedRollout
from utils import EnvState, get_noop_action, save_video, make_small_change, find_latest_checkpoint, load_cpu_config, \
    unwrap_to, OrnsteinUhlenbeckActionNoise
from wrappers.util_wrappers import ResetMode


def save_global_variables():
    d = {}
    for k, v in global_variables.__dict__.items():
        if k.startswith('_'):
            continue
        if k[0].isupper():
            continue
        if 'lock' in str(v).lower():
            continue
        d[k] = v
    return d


def restore_global_variables(d):
    for k, v in d.items():
        setattr(global_variables, k, v)


class RolloutWorker:
    def __init__(self, make_policy_fn_pickle, log_dir, env_state_queue, rollout_queue, worker_n, gv_dict, log_actions,
                 action_shape):
        # Workers shouldn't take up precious GPU memory
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        load_cpu_config(log_dir, 'rollouters')

        np.random.seed(worker_n)
        # Important so that different workers get different actions from env.action_space.sample()
        gym.spaces.prng.seed(worker_n)

        restore_global_variables(gv_dict)
        # Since we're in our own process, this lock isn't actually needed,
        # but other stuff expects this to be initialised
        global_variables.env_creation_lock = multiprocessing.Lock()
        make_policy_fn = cloudpickle.loads(make_policy_fn_pickle)
        # Each worker should seed its policy differently so that when we're doing multiple rollouts from one policy
        # with rollout_randomness=sample, we really do get different rollouts from each worker
        self.policy = make_policy_fn(name='rolloutworker', seed=worker_n)
        self.worker_n = worker_n
        self.env_state_queue = env_state_queue
        self.rollout_queue = rollout_queue
        self.checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        self.rollouts_dir = os.path.join(log_dir, 'demonstrations')
        self.last_action = None
        self.log_dir = log_dir
        self.log_actions = log_actions
        env = None

        mu = np.zeros(action_shape)
        assert isinstance(global_variables.rollout_noise_sigma, float)
        sigma = global_variables.rollout_noise_sigma * np.ones(action_shape)
        self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=mu, sigma=sigma, seed=None)

        while True:
            policy_name, env_state, noise, rollout_len_frames, show_frames, \
                group_serial, deterministic, noop_actions = env_state_queue.get()

            if env is not None:
                if hasattr(env.unwrapped, 'close'):
                    # MuJoCo environments need to have the viewer closed.
                    # Otherwise it leaks memory.
                    env.unwrapped.close()
            env = env_state.env

            if policy_name == 'redo':
                self.redo_rollout(env, env_state, group_serial)
            else:
                self.load_policy_checkpoint(policy_name)
                self.rollout(policy_name, env, env_state.obs, group_serial,
                             noise, rollout_len_frames, show_frames, deterministic, noop_actions)
            # is sometimes slow to flush in processes; be more proactive so output is less confusing
            sys.stdout.flush()

    def load_policy_checkpoint(self, policy_name):
        while True:
            try:
                ckpt_path = os.path.join(self.checkpoint_dir, 'policy-{}-'.format(policy_name))
                last_ckpt_name = find_latest_checkpoint(ckpt_path)
                self.policy.load_checkpoint(last_ckpt_name)
            except NotFoundError:
                # If e.g. the checkpoint got replaced with a newer one
                print("Warning: while loading rollout policy checkpoint: not found. Trying again")
                time.sleep(0.5)
            except Exception as e:
                print("Warning: while loading rollout policy checkpoint:", e, "- trying again")
                time.sleep(0.5)
            else:
                break

    def redo_rollout(self, env, env_state, group_serial):
        # noinspection PyTypeChecker
        rollout = CompressedRollout(final_env_state=env_state,
                                    obses=None,
                                    frames=None,
                                    actions=None,
                                    rewards=[0.0],  # Needed by oracle
                                    generating_policy='redo')

        rollout_hash = str(rollout.hash)
        rollout.vid_filename = rollout_hash + '.mp4'
        vid_path = os.path.join(self.rollouts_dir, rollout.vid_filename)
        frame = np.zeros_like(env.render(mode='rgb_array'))
        save_video([frame], vid_path)

        with open(os.path.join(self.rollouts_dir, rollout_hash + '.pkl'), 'wb') as f:
            pickle.dump(rollout, f)

        self.rollout_queue.put((group_serial, rollout_hash))

    def rollout(self, policy_name, env, obs, group_serial, noise, rollout_len_frames,
                show_frames, deterministic, noop_actions):
        obses = []
        frames = []
        actions = []
        rewards = []
        done = False
        self.last_action = None
        for _ in range(rollout_len_frames):
            obses.append(np.copy(obs))
            frames.append(env.render(mode='rgb_array'))
            if noop_actions:
                action = np.zeros(env.action_space.shape)
            else:
                action = self.policy.step(obs, deterministic=deterministic)
            if noise:
                action = self.add_noise_to_action(action, env)
            actions.append(action)
            obs, reward, done, info = env.step(action)
            # Fetch environments return numpy float reward which is not serializable
            # float(r) -> convert to native Python float
            reward = float(reward)
            rewards.append(reward)
            if done:
                break
        assert len(obses) == len(frames) == len(actions) == len(rewards)

        # Ensure rollouts don't get the same hash even if they're the same
        frames = make_small_change(frames)

        obses = obses[-show_frames:]
        frames = frames[-show_frames:]
        actions = actions[-show_frames:]
        rewards = rewards[-show_frames:]

        while len(obses) < show_frames:  # if done
            obses.append(obses[-1])
            frames.append(frames[-1])
            actions.append(get_noop_action(env))
            rewards.append(0.0)

        name = policy_name
        if noise:
            name += '-noise'
        rollout = CompressedRollout(final_env_state=EnvState(env, obs, done),
                                    obses=obses,
                                    frames=frames,
                                    actions=actions,
                                    rewards=rewards,
                                    generating_policy=name)
        rollout_hash = str(rollout.hash)

        if 'Seaquest' in str(env):
            # Sometimes a segment will end in a bad place.
            # We allow the oracle to detect this by running the policy for a few extra frames
            # and appending the rewards.
            for _ in range(5):
                action = self.policy.step(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                rollout.rewards = rollout.rewards + [reward]

        rollout.vid_filename = rollout_hash + '.mp4'
        vid_path = os.path.join(self.rollouts_dir, rollout.vid_filename)
        save_video(rollout.frames, vid_path)

        with open(os.path.join(self.rollouts_dir, rollout_hash + '.pkl'), 'wb') as f:
            pickle.dump(rollout, f)

        self.rollout_queue.put((group_serial, rollout_hash))

    def add_noise_to_action(self, action, env):
        assert env.action_space.dtype in [np.int64, np.float32]
        if env.action_space.dtype == np.float32:
            noise = self.ou_noise()
            action += noise
            action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
            if self.log_actions:
                self.append_action_to_log(noise, action)
        elif env.action_space.dtype == np.int64:
            if np.random.rand() < global_variables.rollout_random_action_prob:
                if global_variables.rollout_randomness == RolloutRandomness.RANDOM_ACTION:
                    action = env.action_space.sample()
                elif global_variables.rollout_randomness == RolloutRandomness.CORRELATED_RANDOM_ACTION:
                    if self.last_action and np.random.rand() < global_variables.rollout_random_correlation:
                        action = self.last_action
                    else:
                        action = env.action_space.sample()
                else:
                    raise Exception("Invalid noise mode " + str(global_variables.rollout_randomness))
                self.last_action = action
            else:
                self.last_action = None
        return action

    def append_action_to_log(self, noise, action):
        log_file = os.path.join(self.log_dir, f'actions_worker_{self.worker_n}.log')
        np.set_printoptions(precision=3, floatmode='fixed', sign=' ', suppress=True)
        with open(log_file, 'a') as f:
            print(f"{noise}", file=f)


class PolicyRollouter:
    cur_rollouts: Dict[str, CompressedRollout]

    def __init__(self, env, save_dir,
                 reset_state_queue_in: multiprocessing.Queue,
                 reset_mode_value: multiprocessing.Value,
                 log_dir, make_policy_fn, redo_policy, noisy_policies,
                 rollout_len_seconds, show_from_end_seconds,
                 log_actions=False):
        self.env = env
        self.save_dir = save_dir
        self.reset_state_queue = reset_state_queue_in
        self.reset_mode_value = reset_mode_value
        self.redo_policy = redo_policy
        self.noisy_policies = noisy_policies
        self.rollout_len_seconds = rollout_len_seconds
        if show_from_end_seconds is None:
            show_from_end_seconds = rollout_len_seconds
        self.show_from_end_seconds = show_from_end_seconds
        self.noop_actions = True

        # 'spawn' -> start a fresh process
        # (TensorFlow is not fork-safe)
        self.ctx = multiprocessing.get_context('spawn')
        self.env_state_queue = self.ctx.Queue()
        self.rollout_queue = self.ctx.Queue()
        gv = save_global_variables()
        for n in range(4):
            proc = self.ctx.Process(target=RolloutWorker,
                                    args=(cloudpickle.dumps(make_policy_fn),
                                          log_dir,
                                          self.env_state_queue,
                                          self.rollout_queue,
                                          n,
                                          gv,
                                          log_actions,
                                          env.action_space.shape))
            proc.start()
            global_variables.pids_to_proc_names[proc.pid] = f'rollout_worker_{n}'

    def generate_rollouts_from_reset(self, policies):
        env_state = self.get_reset_state()
        group_serial = self.generate_rollout_group(env_state, 'dummy_last_policy_name', policies, False)
        return group_serial

    def generate_rollout_group(self, env_state: EnvState, last_policy_name, policy_names, force_reset):
        rollout_hashes = []
        if env_state.done or force_reset:
            env_state = self.get_reset_state(env_state)
        group_serial = str(time.time())
        n_rollouts = 0
        rollout_len_frames = int(self.rollout_len_seconds * ROLLOUT_FPS)
        show_frames = int(self.show_from_end_seconds * ROLLOUT_FPS)

        if global_variables.rollout_mode == RolloutMode.PRIMITIVES:
            self.noop_actions = False
            for policy_name in policy_names:
                add_extra_noise = (last_policy_name == 'redo')
                if 'LunarLander' in str(self.env):
                    deterministic = False  # Lunar Lander primitives don't work well if deterministic
                else:
                    deterministic = (policy_name != 'random')
                self.env_state_queue.put((policy_name, env_state, add_extra_noise,
                                          rollout_len_frames, show_frames, group_serial, deterministic,
                                          self.noop_actions))
                n_rollouts += 1
        elif global_variables.rollout_mode == RolloutMode.CUR_POLICY:
            if global_variables.rollout_randomness == RolloutRandomness.SAMPLE_ACTION:
                deterministic = False
                add_extra_noise = False
            elif global_variables.rollout_randomness == RolloutRandomness.RANDOM_ACTION or \
                    global_variables.rollout_randomness == RolloutRandomness.CORRELATED_RANDOM_ACTION:
                deterministic = True
                add_extra_noise = True
            else:
                raise Exception('Invalid rollout randomness', global_variables.rollout_randomness)
            for _ in range(global_variables.n_cur_policy):
                self.env_state_queue.put((policy_names[0], env_state, add_extra_noise,
                                          rollout_len_frames, show_frames, group_serial, deterministic,
                                          self.noop_actions))
                n_rollouts += 1
            # Also add a trajectory sampled directly from the policy
            add_extra_noise = False
            deterministic = True
            self.env_state_queue.put((policy_names[0], env_state, add_extra_noise,
                                      rollout_len_frames, show_frames, group_serial, deterministic,
                                      self.noop_actions))
            n_rollouts += 1
        else:
            raise Exception("Invalid rollout mode", global_variables.rollout_mode)

        if self.redo_policy:
            self.env_state_queue.put(('redo', env_state, None, None, None, group_serial))
            n_rollouts += 1

        while len(rollout_hashes) < n_rollouts:
            group_serial_got, rollout_hash = self.rollout_queue.get()
            if group_serial_got == group_serial:
                rollout_hashes.append(rollout_hash)
            else:  # this rollout belongs to another trajectory concurrently being generated
                self.rollout_queue.put((group_serial_got, rollout_hash))
        self.save_metadata(rollout_hashes, group_serial)

        return group_serial

    def save_metadata(self, rollout_hashes, group_serial):
        filename = 'metadata_' + group_serial + '.json'
        path = os.path.join(self.save_dir, filename)

        # This needs to be done atomically because the web app thread will constantly be checking for new
        # metadata files and will be upset if it finds an empty one
        with atomic_write.atomic_write(path) as f:
            json.dump(rollout_hashes, f)
        print(f"Wrote rollout group '{filename}'")

    def get_reset_state(self, env_state=None):
        if self.reset_mode_value.value == ResetMode.USE_ENV_RESET.value:
            if env_state is None:
                # I once saw a bug where the Atari emulator would get into a bad state, giving an
                # illegal instruction error and then the game crashing. I never figured out exactly
                # where it was, but the error message seemed to come from around the time the
                # environment is reset. Maybe there are problems with multithreaded reset?
                # As a hacky fix, we protect env reset, just in case.
                global_variables.env_creation_lock.acquire()
                obs = self.env.reset()
                global_variables.env_creation_lock.release()
                reset_state = EnvState(self.env, obs, done=False)
            else:
                env = env_state.env
                obs = env.reset()
                reset_state = EnvState(env, obs, done=False)
            return reset_state
        elif self.reset_mode_value.value == ResetMode.FROM_STATE_CACHE.value:
            while True:
                try:
                    reset_state = self.reset_state_queue.get(block=True, timeout=0.1)  # type: EnvState
                    print("Demonstrating from state", reset_state.step_n)
                    break
                except queue.Empty:
                    print("Waiting for demonstrations reset state...")
                    time.sleep(1.0)
            env = reset_state.env
            time_limit = unwrap_to(env, TimeLimit)
            time_limit._episode_started_at = time.time()
            time_limit._elapsed_steps = 0
            reset_state = EnvState(env, reset_state.obs, reset_state.done, reset_state.step_n, reset_state.birthtime)
            return reset_state
        else:
            raise Exception("Invalid demonstration reset mode:", self.reset_mode_value.value)
