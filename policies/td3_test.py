#!/usr/bin/env python3

import copy
import os
import sys
import tempfile
import unittest

import gym
import numpy as np
import tensorflow as tf
from gym.wrappers import FlattenDictWrapper, Monitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from policies.base_policy import PolicyTrainMode
from policies.td3 import TD3Policy, DemonstrationsBuffer
from subproc_vec_env_custom import CustomDummyVecEnv, CustomSubprocVecEnv
from wrappers.fetch_pick_and_place_register import register
from wrappers.fetch_pick_and_place import RandomInitialPosition
from wrappers.util_wrappers import LogEpisodeStats

tf.logging.set_verbosity(tf.logging.ERROR)
register()


class Oracle:
    def __init__(self, mode):
        self.gripping = None
        self.step_n = None
        self.last_action = None
        self.mode = mode

    def reset(self):
        self.gripping = False
        self.step_n = 0

    def get_action(self, obs):
        if self.mode == 'smooth':
            return self.get_action_smooth(obs)
        elif self.mode == 'zigzag':
            return self.get_action_zigzag(obs)
        else:
            raise Exception()

    def get_action_smooth(self, obs):
        gripper_to_block = obs[:3]
        block_to_target = obs[3:6]
        gripper_width = obs[6]
        if np.linalg.norm(gripper_to_block) > 0.01 and not self.gripping:
            if np.linalg.norm(gripper_to_block[:2]) > 0.05:
                z = 0
            else:
                z = gripper_to_block[2]
            action = np.concatenate([3 * gripper_to_block[:2], [3 * z, 1]])
        elif gripper_width > 0.05:
            action = np.array([0, 0, 0, -1])
        else:
            self.gripping = True
            action = np.concatenate([3 * block_to_target, [-1]])
        return action

    def get_action_zigzag(self, obs):
        if self.step_n % 5 != 0:
            self.step_n += 1
            return self.last_action

        assert obs.shape == (7,)
        gripper_to_block = obs[:3]
        block_to_target = obs[3:6]
        gripper_width = obs[6]
        if np.linalg.norm(gripper_to_block) > 0.03 and not self.gripping:
            di = np.argmax(np.abs(gripper_to_block))
            action = np.array([0., 0., 0., 1.])
            action[di] = 0.2 * np.sign(gripper_to_block[di])
        elif gripper_width > 0.05:
            action = np.array([0, 0, 0, -1])
        else:
            self.gripping = True
            di = np.argmax(np.abs(block_to_target))
            action = np.array([0., 0., 0., -1.])
            action[di] = 0.2 * np.sign(block_to_target[di])

        self.last_action = action
        self.step_n += 1
        return action


def gen_demonstrations(env_id, log_dir, n_demonstrations, demonstrations_buffer: DemonstrationsBuffer, oracle):
    env = gym.make(env_id)
    env.seed(0)
    np.random.seed(0)
    # env = Monitor(env, video_callable=lambda n: True, directory=log_dir, uid=111)
    env = RandomInitialPosition(env)
    env = LogEpisodeStats(env, log_dir, '_demo')

    for n in range(n_demonstrations):
        print(f"Generating demonstration {n}...")
        obs, done = env.reset(), False
        oracle.reset()
        while not done:
            action = oracle.get_action(obs)
            demonstrations_buffer.store(obs, action)
            obs, reward, done, info = env.step(action)


def get_replay_buffer(env, env_id):
    policy = TD3Policy('dummyname',
                       env_id,
                       env.observation_space,
                       env.action_space,
                       n_envs=1,
                       rollouts_per_worker=2,
                       batch_size=256,
                       cycles_per_epoch=50,
                       batches_per_cycle=40,
                       noise_sigma=0.2,
                       polyak=0.995,
                       n_initial_episodes=3)
    policy.set_training_env(env)
    policy.init_logger(tempfile.mkdtemp())
    while policy.initial_exploration_phase:
        policy.train()
    return copy.deepcopy(policy.replay_buffer)  # Should end at about -1.7 AverageTestEpRet


class TestTD3(unittest.TestCase):
    @staticmethod
    def env_fn(seed, env_id):
        env = gym.make(env_id)
        if 'Fetch' in env.spec.id:
            env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
        env.seed(seed)
        return env

    spinningup_hyperparams = dict(
        rollouts_per_worker=1, hidden_sizes=(300,), batch_size=100, cycles_per_epoch=5, n_initial_episodes=10,
        batches_per_cycle=1000, l2_coef=0.0, noise_type='gaussian', polyak=0.995
    )

    fetch_hyperparams = dict()

    def _test_cheetah_single(self):
        test_ret = self.run_td3_rl('HalfCheetah-v2', n_envs=1, n_epochs=10, hyperparams=self.spinningup_hyperparams)
        self.assertGreater(test_ret, 2000)

    def test_cheetah_parallel(self):
        test_ret = self.run_td3_rl('HalfCheetah-v2', n_envs=10, n_epochs=10, hyperparams=self.spinningup_hyperparams)
        self.assertGreater(test_ret, 2500)

    def _test_reach_single(self):
        test_ret = self.run_td3_rl('FetchReachDense-v1', n_envs=1, n_epochs=5, hyperparams=self.fetch_hyperparams)
        self.assertGreater(test_ret, -1.0)

    def test_reach_parallel(self):
        test_ret = self.run_td3_rl('FetchReachDense-v1', n_envs=16, n_epochs=2, hyperparams=self.fetch_hyperparams)
        self.assertGreater(test_ret, -1.0)

    def run_td3_rl(self, env_id, n_envs, n_epochs, hyperparams):
        tmp_dir = tempfile.mkdtemp()
        print("Logging to", tmp_dir)
        train_env = CustomSubprocVecEnv(env_fns=[lambda env_n=env_n: self.env_fn(env_id=env_id, seed=env_n)
                                                 for env_n in range(n_envs)])
        test_env = self.env_fn(env_id=env_id, seed=n_envs)
        # test_env = Monitor(test_env, tmp_dir, video_callable=lambda n: True)

        # Increase chance of TensorFlow being deterministic
        # https://stackoverflow.com/a/39938524/7832197
        config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        policy = TD3Policy('dummyname',
                           env_id,
                           train_env.observation_space,
                           train_env.action_space,
                           n_envs=n_envs, train_mode=PolicyTrainMode.R_ONLY,
                           sess_config=config,
                           **hyperparams)

        policy.init_logger(tmp_dir)
        policy.set_training_env(train_env)
        policy.test_env = test_env
        last_epoch_n = 0
        test_return = None
        while policy.epoch_n <= n_epochs:
            policy.train()
            if policy.epoch_n != last_epoch_n:
                print("Epoch", policy.epoch_n - 1)
                print("  Total steps:", policy.n_serial_steps)
                test_return = np.mean(policy.test_agent(n=5))
                print("  Average test return:", test_return)
                last_epoch_n = policy.epoch_n
                with policy.graph.as_default():
                    var_sum = policy.sess.run(tf.reduce_sum([tf.reduce_sum(v) for v in tf.trainable_variables()]))
                    print("  Policy hash:", var_sum)
            sys.stdout.flush()
        train_env.close()
        test_env.close()

        return test_return

    def _test_pickplace_smooth_demonstrations(self):
        oracle = Oracle('smooth')
        self.run_td3_bc(oracle)

    def test_pickplace_zigzag_demonstrations(self):
        oracle = Oracle('zigzag')
        self.run_td3_bc(oracle)

    def run_td3_bc(self, oracle):
        env_id = 'FetchPickAndPlace-Repeat1-ContGripper-WithGripObs-5InitialBlockPos-FixedGoal-Delta-GripperBonuses-v0'
        temp_dir = tempfile.mkdtemp()
        print("Logging to", temp_dir)
        n_demos = 100

        dummy_env = gym.make(env_id)
        obs_space = dummy_env.observation_space
        act_space = dummy_env.action_space

        def make_policy():
            return TD3Policy('dummyname',
                             env_id,
                             obs_space,
                             act_space,
                             n_envs=1,
                             hidden_sizes=(256, 256, 256, 256))

        policy = make_policy()

        test_env = gym.make(env_id)
        test_env.seed(0)
        test_env = Monitor(test_env, directory=temp_dir, video_callable=lambda n: True)
        test_env = LogEpisodeStats(test_env, log_dir=temp_dir, stdout=False)
        policy.test_env = test_env

        policy.init_logger(temp_dir)
        gen_demonstrations(env_id, os.path.join(temp_dir, 'demos'), n_demos, policy.demonstrations_buffer, oracle)
        last_epoch_n = None
        while policy.epoch_n < 10:
            if policy.epoch_n != last_epoch_n:
                last_epoch_n = policy.epoch_n
                print(f"Epoch {last_epoch_n}")
            print(f"Cycle {policy.cycle_n}")
            policy.train_bc()
        policy.test_agent(n=30)
        self.assertGreater(test_env.stats['success_rate'], 0.6)


class TestReplayBufferVecEnv(unittest.TestCase):
    """
    Test whether we get exactly the same replay buffer when using our
    hacked-up SubprocVecEnv as when using just a single environment
    """

    def test_custom_vecenv(self):
        env_id = 'FetchPickAndPlace-v1'

        def env_fn():
            env = gym.make(env_id)
            env.seed(0)
            env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
            return env

        # Check 1: make sure we get deterministic results
        replay_buffers_correct = []
        for i in range(2):
            env = CustomDummyVecEnv(env_fn())
            replay_buffers_correct.append(get_replay_buffer(env, env_id))
            env.close()
        self.compare_buffers(replay_buffers_correct[0], replay_buffers_correct[1])

        # Check 2: confirm that the replay buffer using SubprocVecEnv is exactly the same
        env = CustomSubprocVecEnv([env_fn])
        replay_buffer_test = get_replay_buffer(env, env_id)
        self.compare_buffers(replay_buffers_correct[0], replay_buffer_test)
        env.close()

    def compare_buffers(self, b1, b2):
        for k in vars(b1):
            v1 = vars(b1)[k]
            v2 = vars(b2)[k]
            if isinstance(v1, np.ndarray):
                np.testing.assert_array_almost_equal(v1, v2)
            elif isinstance(v1, int):
                self.assertEqual(v1, v2)
            else:
                raise Exception()


if __name__ == '__main__':
    unittest.main()
