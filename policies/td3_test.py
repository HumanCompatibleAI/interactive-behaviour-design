import unittest

import numpy as np

from policies.td3 import Batch, combine_batches


class Test(unittest.TestCase):
    def test_combine_batches(self):
        b1 = Batch(
            obs1=[0],
            obs2=[1],
            acts=[2],
            rews=[3],
            done=[4],
        )
        b2 = Batch(
            obs1=[10],
            obs2=[11],
            acts=[12],
            rews=[13],
            done=[14],
        )
        b = combine_batches(b1, b2)
        np.testing.assert_array_equal(b.obs1, [0, 10])
        np.testing.assert_array_equal(b.obs2, [1, 11])
        np.testing.assert_array_equal(b.acts, [2, 12])
        np.testing.assert_array_equal(b.rews, [3, 13])
        np.testing.assert_array_equal(b.done, [4, 14])

        shape = [2, 3, 5]
        b1 = Batch(
            obs1=np.zeros(shape),
            obs2=np.zeros(shape),
            acts=np.zeros(shape),
            rews=np.zeros(shape),
            done=np.zeros(shape),
        )
        b2 = Batch(
            obs1=np.zeros(shape),
            obs2=np.zeros(shape),
            acts=np.zeros(shape),
            rews=np.zeros(shape),
            done=np.zeros(shape),
        )
        b = combine_batches(b1, b2)

        for v in [v for v in dir(b) if '__' not in v and v != 'len']:
            expected_len = getattr(b1, v).shape[0] + getattr(b2, v).shape[0]
            expected_shape = (expected_len,) + getattr(b1, v).shape[1:]
            assert getattr(b, v).shape == expected_shape


if __name__ == '__main__':
    unittest.main()
