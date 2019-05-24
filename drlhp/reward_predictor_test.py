import os
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf

sys.path.insert(0, '..')

from drlhp.reward_predictor_core_network import net_mlp, net_cnn
from drlhp.pref_db import PrefDB
from drlhp.reward_predictor import RewardPredictor, MIN_L2_REG_COEF

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


if __name__ == '__main__':
    unittest.main()
