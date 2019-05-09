import json
import multiprocessing
import sys

import os
import pickle
import queue
import time
from typing import Dict

import numpy as np
from cloudpickle import cloudpickle
from gym.utils import atomic_write
from tensorflow.python.framework.errors_impl import NotFoundError

import global_variables
from global_variables import RolloutMode, RolloutRandomness
from global_constants import ROLLOUT_FPS
from rollouts import CompressedRollout
from utils import EnvState, get_noop_action, save_video, make_small_change, \
    find_latest_checkpoint, set_env_state
from wrappers.util_wrappers import ResetMode


class RolloutWorker:
    def __init__(self, make_policy_fn_pickle, log_dir, env_state_queue, rollout_queue, worker_n):
        # Workers shouldn't take up precious GPU memory
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        # Since we're in our own process, this lock isn't actually needed,
        # but other stuff expects this to be initialised
        global_variables.env_creation_lock = multiprocessing.Lock()
        make_policy_fn = cloudpickle.loads(make_policy_fn_pickle)
        # Each worker should seed its policy differently so that when we're doing multiple rollouts from one policy
        # with rollout_randomness=sample, we really do get different rollouts from each worker
        self.policy = make_policy_fn(name='rolloutworker', seed=worker_n)
        self.env_state_queue = env_state_queue
        self.rollout_queue = rollout_queue
        self.checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        self.rollouts_dir = os.path.join(log_dir, 'demonstrations')
        env = None

        while True:
            policy_name, env_state, noise, rollout_len_frames, show_frames, group_serial, deterministic = env_state_queue.get()

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
                             noise, rollout_len_frames, show_frames, deterministic)
            # is sometimes slow to flush in processes; be more proactive so output is less confusing
            sys.stdout.flush()

    def load_policy_checkpoint(self, policy_name):
        while True:
            try:
                last_ckpt_name = find_latest_checkpoint(self.checkpoint_dir,
                                                        'policy-{}-'.format(policy_name))
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

    def rollout(self, policy_name, env, obs, group_serial, noise, rollout_len_frames, show_frames, deterministic):
        obses = []
        frames = []
        actions = []
        rewards = []
        done = False
        for _ in range(rollout_len_frames):
            obses.append(np.copy(obs))
            frames.append(env.render(mode='rgb_array'))
            action = self.policy.step(obs, deterministic=deterministic)
            if noise:
                assert env.action_space.dtype in [np.int64, np.float32]
                if env.action_space.dtype == np.float32:
                    action += 0.3 * env.action_space.sample()
                elif env.action_space.dtype == np.int64 and np.random.rand() < global_variables.rollout_action_noise:
                    action = env.action_space.sample()
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

class PolicyRollouter:
    cur_rollouts: Dict[str, CompressedRollout]

    def __init__(self, env, save_dir,
                 reset_state_queue_in: multiprocessing.Queue,
                 reset_mode_value: multiprocessing.Value,
                 log_dir, make_policy_fn, redo_policy, noisy_policies,
                 rollout_len_seconds, show_from_end_seconds):
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

        # 'spawn' -> start a fresh process
        # (TensorFlow is not fork-safe)
        self.ctx = multiprocessing.get_context('spawn')
        self.env_state_queue = self.ctx.Queue()
        self.rollout_queue = self.ctx.Queue()
        for n in range(16):
            self.ctx.Process(target=RolloutWorker, args=(cloudpickle.dumps(make_policy_fn),
                                                         log_dir,
                                                         self.env_state_queue,
                                                         self.rollout_queue,
                                                         n)).start()

    def generate_rollouts_from_reset(self, policies, softmax_temp):
        env_state = self.get_reset_state()
        group_serial = self.generate_rollout_group(env_state, 'dummy_last_policy_name', policies, softmax_temp, False)
        return group_serial

    def generate_rollout_group(self, env_state: EnvState, last_policy_name, policy_names,
                               softmax_temp, force_reset):
        rollout_hashes = []
        if env_state.done or force_reset:
            env_state = self.get_reset_state(env_state)
        group_serial = str(time.time())
        n_rollouts = 0
        rollout_len_frames = int(self.rollout_len_seconds * ROLLOUT_FPS)
        show_frames = int(self.show_from_end_seconds * ROLLOUT_FPS)

        if global_variables.rollout_mode == RolloutMode.primitives:
            for policy_name in policy_names:
                noise = (last_policy_name == 'redo')
                if 'LunarLander' in str(self.env):
                    deterministic = False  # Lunar Lander primitives don't work well if deterministic
                else:
                    deterministic = (policy_name != 'random')
                self.env_state_queue.put((policy_name, env_state, noise,
                                          rollout_len_frames, show_frames, group_serial, deterministic))
                n_rollouts += 1
        elif global_variables.rollout_mode == RolloutMode.cur_policy:
            if global_variables.rollout_randomness == RolloutRandomness.sample:
                deterministic = False
                noise = False
            elif global_variables.rollout_randomness == RolloutRandomness.noise:
                deterministic = True
                noise = True
            else:
                raise Exception('Invalid rollout randomness', global_variables.rollout_randomness)
            for _ in range(global_variables.n_cur_policy):
                self.env_state_queue.put((policy_names[0], env_state, noise,
                                          rollout_len_frames, show_frames, group_serial, deterministic))
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
                    reset_state = self.reset_state_queue.get(block=True,
                                                             timeout=0.1)  # type: EnvState
                    print("Demonstrating from state", reset_state.step_n)
                    break
                except queue.Empty:
                    print("Waiting for demonstrations reset state...")
                    time.sleep(1.0)
            return reset_state
        else:
            raise Exception("Invalid demonstration reset mode:", self.reset_mode_value.value)
