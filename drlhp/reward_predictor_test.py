import glob
import os
import sys
import tempfile
import unittest
from typing import List

import numpy as np
import tensorflow as tf
from gym.spaces import Box
from matplotlib.pyplot import plot, legend, show, grid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from drlhp.reward_predictor_core_network import net_mlp, net_cnn
from drlhp.pref_db import PrefDB
from drlhp.reward_predictor import RewardPredictor, MIN_L2_REG_COEF, PredictedRewardNormalization
import global_variables
import throttler
from utils import ObsRewardTuple, load_reference_trajectory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

throttler.throttle_sleep = lambda event_type: None
throttler.mark_event = lambda event_type: None


# Run me with
#   coverage run --include reward_predictor.py reward_predictor_test.py SmokeTests
# then
#   coverage html
class SmokeTests(unittest.TestCase):
    def test(self):
        global_variables.predicted_rewards_normalize_mean_std = '0,1'
        global_variables.log_reward_normalization_every_n_calls = 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            for net, args in [(net_mlp, {}), (net_cnn, {'dropout': 0.0, 'batchnorm': False})]:
                if net == net_cnn:
                    obs_space = Box(low=-1, high=1, shape=(84, 84, 4), dtype=np.float32)
                else:
                    obs_space = Box(low=np.array([-1]), high=np.array([+1]), dtype=np.float32)
                rp = RewardPredictor(obs_space=obs_space,
                                     network=net,
                                     network_args=args,
                                     r_std=0.1,
                                     log_dir=tmp_dir,
                                     seed=0,
                                     name='test',
                                     reward_normalization=PredictedRewardNormalization.OFF,
                                     normalization_loss_coef=0,
                                     )
            obs_space = Box(low=np.array([-1]), high=np.array([+1]), dtype=np.float32)
            obses = np.random.rand(*((2,) + obs_space.shape))
            for reward_normalization in PredictedRewardNormalization:
                rp = RewardPredictor(obs_space=obs_space,
                                     network=net_mlp,
                                     network_args={},
                                     r_std=0.1,
                                     log_dir=tmp_dir,
                                     seed=0,
                                     name='test',
                                     reward_normalization=reward_normalization,
                                     normalization_loss_coef=0,
                                     )
                rp.unnormalized_rewards(obses)
                for update_normalization in [False, True]:
                    rp.normalized_rewards(obses[np.newaxis, 0], update_normalization=update_normalization)
                    rp.normalized_rewards(obses, update_normalization=update_normalization)

            prefs_train = PrefDB(maxlen=100)
            prefs_train.append(np.array([obses[0]]), np.array([obses[1]]), [1, 0])
            prefs_val = PrefDB(maxlen=100)
            prefs_val.append(np.array([obses[0]]), np.array([obses[1]]), [1, 0])
            for verbose in [True, False]:
                for reward_normalization in PredictedRewardNormalization:
                    rp.reward_normalization = reward_normalization
                    rp.train(prefs_train, prefs_val, val_interval=1, verbose=verbose)
            for _ in range(100):
                obses = np.random.rand(*((2,) + obs_space.shape))
                obs1 = np.array([obses[0]])
                obs2 = np.array([obses[1]])
                prefs_train.append(obs1, obs2, [1, 0])
                prefs_val.append(obs1, obs2, [1, 0])
            rp.train(prefs_train, prefs_val, val_interval=1)
            ckpt_name = os.path.join(tmp_dir, 'ckpt')
            for max_to_keep in [1, None]:
                rp.save(ckpt_name, max_to_keep=max_to_keep)
                rp.save(ckpt_name, max_to_keep=max_to_keep)
            c = rp.get_latest_checkpoint(ckpt_name)
            rp.load(c)
            rp.load(c, polyak_coef=0)
            rp.reset_normalisation()

            for _ in range(100):
                rp.train(prefs_train, prefs_val, val_interval=1, verbose=verbose)


class TestRewardNormalization(unittest.TestCase):
    def _test(self):
        ckpt_path = '/private/var/folders/z2/5bbsrgpj7y51rhv12n7w9q9h0000gn/T/tmp.LDcnFZqR/reward_predictor.ckpt.84'
        obs_space = Box(low=float('-inf'), high=float('inf'), shape=(6,), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp_dir:
            global_variables.log_reward_normalization_every_n_calls = 1000
            global_variables.predicted_rewards_normalize_min_max = '-3.83,-0.05'
            rp = RewardPredictor(obs_space=obs_space,
                                 network=net_mlp,
                                 network_args={},
                                 r_std=1.0,
                                 log_dir=tmp_dir,
                                 seed=0,
                                 reward_normalization=PredictedRewardNormalization.EXTREME_TRAINING_STATES,
                                 normalization_loss_coef=0,
                                 name='test')
            rp.load(ckpt_path)

            env_id = 'FetchReach-CustomActionRepeat5ActionLimit0.2-v0'
            reference_trajectory = load_reference_trajectory(env_id)  # type: List[ObsRewardTuple]
            obses = np.array([tup.obs for tup in reference_trajectory])
            env_rewards = [tup.reward for tup in reference_trajectory]
            print("Environment env_rewards range, scale:", np.min(env_rewards), np.max(env_rewards))

            pred_rewards = rp.normalized_rewards(obses)
            print("Predicted pred_rewards mean, std:", np.min(pred_rewards), np.max(env_rewards))

            plot(env_rewards, label='Environment rewards')
            plot(pred_rewards, label='Predicted rewards')
            grid()
            legend()
            show()



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
