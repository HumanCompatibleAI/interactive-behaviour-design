import pickle
import random
import unittest
from math import ceil

import numpy as np

from basicfetch.baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise as DDPGOrnsteinUhlenbeckActionNoise
from utils import batch_iter, RunningProportion
from utils import OrnsteinUhlenbeckActionNoise as UtilsOrnsteinUhlenbeckActionNoise


class TestUtils(unittest.TestCase):
    def test_batch_iter_1(self):
        """
        Check that batch_iter gives us exactly the right data back.
        """
        l1 = list(range(16))
        l2 = list(range(15))
        l3 = list(range(13))
        for l in [l1, l2, l3]:
            for shuffle in [True, False]:
                expected_data = l
                actual_data = set()
                expected_n_batches = ceil(len(l) / 4)
                actual_n_batches = 0
                for batch_n, x in enumerate(batch_iter(l, batch_size=4, shuffle=shuffle)):
                    if batch_n == expected_n_batches - 1 and len(l) % 4 != 0:
                        self.assertEqual(len(x), len(l) % 4)
                    else:
                        self.assertEqual(len(x), 4)
                    self.assertEqual(len(actual_data.intersection(set(x))), 0)
                    actual_data = actual_data.union(set(x))
                    actual_n_batches += 1
                self.assertEqual(actual_n_batches, expected_n_batches)
                np.testing.assert_array_equal(list(actual_data), expected_data)

    def test_batch_iter_2(self):
        """
        Check that shuffle=True returns the same data but in a different order.
        """
        expected_data = list(range(16))
        actual_data = []
        for x in batch_iter(expected_data, batch_size=4, shuffle=True):
            actual_data.extend(x)
        self.assertEqual(len(actual_data), len(expected_data))
        self.assertEqual(set(actual_data), set(expected_data))
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(actual_data, expected_data)

    def test_batch_iter_3(self):
        """
        Check that successive calls shuffle in a different order.
        """
        data = list(range(16))
        out1 = []
        for x in batch_iter(data, batch_size=4, shuffle=True):
            out1.extend(x)
        out2 = []
        for x in batch_iter(data, batch_size=4, shuffle=True):
            out2.extend(x)
        self.assertEqual(set(out1), set(out2))
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(out1, out2)

    def test_running_proportion(self):
        l = [random.random() < 0.3 for _ in range(10)]
        print(l)
        rp = RunningProportion()
        for i, e in enumerate(l):
            rp.update(e)
            actual = rp.v
            expected = l[:i + 1].count(True) / (i + 1)
            self.assertEqual(actual, expected)

    def test_ornstein_uhlenbeck_correctness(self):
        mu = np.array([0])
        sigma = np.array([0.2])
        np.random.seed(0)
        ou_ddpg = DDPGOrnsteinUhlenbeckActionNoise(mu=mu, sigma=sigma)
        ou_utils = UtilsOrnsteinUhlenbeckActionNoise(mu=mu, sigma=sigma, seed=0)
        for _ in range(10):
            self.assertEqual(ou_utils(), ou_ddpg())

    def test_ornstein_uhlenbeck_pickling(self):
        mu = np.array([0])
        sigma = np.array([0.2])
        ou = UtilsOrnsteinUhlenbeckActionNoise(mu=mu, sigma=sigma)
        # Get to a noise state which is good and random
        for _ in range(10):
            ou()
        ou2 = pickle.loads(pickle.dumps(ou))
        for _ in range(10):
            self.assertEqual(ou2(), ou())


if __name__ == '__main__':
    unittest.main()
