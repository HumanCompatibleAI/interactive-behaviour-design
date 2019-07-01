import glob
import os
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from gym.spaces import Box

sys.path.insert(0, '..')

from drlhp.reward_predictor_core_network import net_mlp, net_cnn
from drlhp.pref_db import PrefDB
from drlhp.reward_predictor import RewardPredictor, MIN_L2_REG_COEF, PredictedRewardNormalization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


class TestRewardPredictor(unittest.TestCase):
    def test_l2_reg_mlp(self):
        self.run_net_l2_test(net_mlp, {}, [10])

    def test_l2_reg_cnn(self):
        self.run_net_l2_test(net_cnn, {'batchnorm': False, 'dropout': 0.0}, [84, 84, 4])

    def run_net_l2_test(self, net, net_args, obs_shape):
        n_steps = 1
        tmp_dir = tempfile.mkdtemp()
        rp = RewardPredictor(obs_shape=obs_shape,
                             network=net,
                             network_args=net_args,
                             r_std=0.1,
                             log_dir=tmp_dir,
                             seed=0,
                             name='test')
        with rp.sess.graph.as_default():
            manual_l2_loss = tf.add_n([tf.norm(v) for v in tf.trainable_variables()])

        prefs_train = PrefDB(maxlen=10)
        s1 = np.random.rand(n_steps, *obs_shape)
        s2 = np.random.rand(n_steps, *obs_shape)
        prefs_train.append(s1, s2, pref=(1.0, 0.0))

        # Test 1: if we turn off L2 regularisation, does the L2 loss go up?
        rp.l2_reg_coef = 0.0
        l2_start = rp.sess.run(manual_l2_loss)
        for _ in range(100):
            rp.train(prefs_train=prefs_train, prefs_val=prefs_train,
                     val_interval=1000, verbose=False)
        l2_end = rp.sess.run(manual_l2_loss)
        # Threshold set empirically while writing test
        self.assertTrue(l2_end > l2_start + 0.1)

        # Test 2: if we turn it back on, does it go down?
        rp.l2_reg_coef = MIN_L2_REG_COEF
        l2_start = rp.sess.run(manual_l2_loss)
        for _ in range(100):
            rp.train(prefs_train=prefs_train, prefs_val=prefs_train,
                     val_interval=1000, verbose=False)
        l2_end = rp.sess.run(manual_l2_loss)
        # Threshold set empirically while writing test
        self.assertTrue(l2_end < l2_start - 0.5)

    def test_save_load(self):
        np.random.seed(0)
        for net, network_args, obs_shape in [(net_mlp, {}, (100,)),
                                             (net_cnn, {'batchnorm': False, 'dropout': 0}, (100, 100, 4))]:
            self._test_save_load_basic(net, network_args, obs_shape)
            self._test_load_latest_checkpoint(net, network_args, obs_shape)
            self._test_remove_old_checkpoints(net, network_args, obs_shape)
            self._test_polyak_save_load(net, network_args, obs_shape)

    def _test_save_load_basic(self, net, network_args, obs_shape):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rp1, rp2 = self.get_reward_predictor_pair(net, network_args, obs_shape, tmp_dir)
            ckpt_path = os.path.join(tmp_dir, 'checkpoint')
            self.check_reward_predictors_different(rp1, rp2)
            self.save_load_reward_predictor(rp1, rp2, ckpt_path)
            self.check_reward_predictors_identical(rp1, rp2)

    def _test_load_latest_checkpoint(self, net, network_args, obs_shape):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rp1, rp2 = self.get_reward_predictor_pair(net, network_args, obs_shape, tmp_dir)
            ckpt_path = os.path.join(tmp_dir, 'checkpoint')
            self.save_load_reward_predictor(rp1, rp2, ckpt_path)
            self.check_reward_predictors_identical(rp1, rp2)

            for _ in range(3):
                self.train_reward_predictor(rp1)
                self.check_reward_predictors_different(rp1, rp2)
                self.save_load_reward_predictor(rp1, rp2, ckpt_path)
                self.check_reward_predictors_identical(rp1, rp2)

    def _test_remove_old_checkpoints(self, net, network_args, obs_shape):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rp = self.get_reward_predictor(net, network_args, obs_shape, tmp_dir, 0)
            ckpt_path = os.path.join(tmp_dir, 'checkpoint')
            rp.save(ckpt_path, max_to_keep=2)
            self.assertEqual(glob.glob(ckpt_path + '.*'), [ckpt_path + '.0'])
            rp.save(ckpt_path, max_to_keep=2)
            self.assertEqual(glob.glob(ckpt_path + '.*'), [ckpt_path + '.0', ckpt_path + '.1'])
            rp.save(ckpt_path, max_to_keep=2)
            self.assertEqual(glob.glob(ckpt_path + '.*'), [ckpt_path + '.1', ckpt_path + '.2'])

    def _test_polyak_save_load(self, net, network_args, obs_shape):
        obs = np.random.random_sample(obs_shape)
        with tempfile.TemporaryDirectory() as tmp_dir:
            rp1, rp2 = self.get_reward_predictor_pair(net, network_args, obs_shape, tmp_dir)
            ckpt_path = os.path.join(tmp_dir, 'checkpoint')

            self.check_reward_predictors_different(rp1, rp2)
            r2_old = self.predict_reward(rp2, obs)
            self.save_load_reward_predictor_polyak(rp1, rp2, ckpt_path, polyak_coef=1.0)
            # With a polyak coefficient of 1.0, we should have kept all the old parameter values,
            # so we should get exactly the old result
            r2_new = self.predict_reward(rp2, obs)
            self.check_reward_predictors_different(rp1, rp2)
            self.assertEqual(r2_old, r2_new)
            r2_old = r2_new

            self.save_load_reward_predictor_polyak(rp1, rp2, ckpt_path, polyak_coef=0.0)
            # This time, we should have loaded exactly the parameter values from the first reward predictor
            r2_new = self.predict_reward(rp2, obs)
            with self.assertRaises(AssertionError):
                self.assertEqual(r2_new, r2_old)
            r1 = self.predict_reward(rp1, obs)
            self.assertEqual(r2_new, r1)
            r2_old = r2_new

            self.train_reward_predictor(rp1)
            self.save_load_reward_predictor_polyak(rp1, rp2, ckpt_path, polyak_coef=0.5)
            # Now we should get something different from rp1, and also different from the old rp2
            r1 = self.predict_reward(rp1, obs)
            r2_new = self.predict_reward(rp2, obs)
            with self.assertRaises(AssertionError):
                self.assertEqual(r2_new, r1)
            with self.assertRaises(AssertionError):
                self.assertEqual(r2_new, r2_old)

            # If we call load on rp2 a bunch of times, it should get closer to rp1
            r1 = self.predict_reward(rp1, obs)
            r2 = self.predict_reward(rp2, obs)
            with self.assertRaises(AssertionError):
                np.testing.assert_approx_equal(r2, r1, significant=5)
            for _ in range(20):
                self.save_load_reward_predictor_polyak(rp1, rp2, ckpt_path, polyak_coef=0.1)
            r2 = self.predict_reward(rp2, obs)
            np.testing.assert_approx_equal(r2, r1, significant=5)

    def get_reward_predictor_pair(self, net, network_args, obs_shape, tmp_dir):
        rp1 = self.get_reward_predictor(net, network_args, obs_shape, tmp_dir, seed=0)
        rp2 = self.get_reward_predictor(net, network_args, obs_shape, tmp_dir, seed=1)
        return rp1, rp2

    def get_reward_predictor(self, net, network_args, obs_shape, tmp_dir, seed):
        obs_space = Box(low=0, high=1, shape=obs_shape)
        return RewardPredictor(obs_space=obs_space, network=net, network_args=network_args, r_std=0.1, seed=seed,
                               log_dir=tmp_dir, name='test',
                               reward_normalization=PredictedRewardNormalization.OFF,
                               normalization_loss_coef=0.9)

    def check_reward_predictors_different(self, rp1, rp2):
        obs = np.random.random_sample(rp1.obs_shape)
        r1 = self.predict_reward(rp1, obs)
        r2 = self.predict_reward(rp2, obs)
        with self.assertRaises(AssertionError):
            self.assertEqual(r1, r2)

    def check_reward_predictors_identical(self, rp1, rp2):
        obs = np.random.random_sample(rp1.obs_shape)
        r1 = self.predict_reward(rp1, obs)
        r2 = self.predict_reward(rp2, obs)
        self.assertEqual(r1, r2)

    def predict_reward(self, reward_predictor, obs):
        return reward_predictor.unnormalized_rewards(np.array([obs]))[0][0]

    def save_load_reward_predictor(self, rp1, rp2, ckpt_path):
        rp1.save(ckpt_path)
        latest_ckpt_path = RewardPredictor.get_latest_checkpoint(ckpt_path)
        rp2.load(latest_ckpt_path)

    def save_load_reward_predictor_polyak(self, rp1, rp2, ckpt_path, polyak_coef):
        rp1.save(ckpt_path)
        latest_ckpt_path = RewardPredictor.get_latest_checkpoint(ckpt_path)
        rp2.load(latest_ckpt_path, polyak_coef=polyak_coef)

    def train_reward_predictor(self, reward_predictor):
        prefs_train, prefs_val = PrefDB(maxlen=10), PrefDB(maxlen=10)
        n_steps = 10
        s1 = np.random.random_sample((n_steps,) + reward_predictor.obs_shape)
        s2 = np.random.random_sample((n_steps,) + reward_predictor.obs_shape)
        prefs_train.append(s1, s2, pref=(1.0, 0.0))
        reward_predictor.train(prefs_train, prefs_val, val_interval=1000)


if __name__ == '__main__':
    unittest.main()
