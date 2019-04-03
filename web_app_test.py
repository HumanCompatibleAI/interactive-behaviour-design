import os
import tempfile
import unittest
from itertools import combinations

from web_app import web_globals


class TestSegmentComparisons(unittest.TestCase):

    def test_sample(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            web_globals.experience_dir = temp_dir
            web_globals._segments_dir = temp_dir

            from web_app.comparisons import sample_seg_pair, mark_compared

            for n in range(3):
                open(os.path.join(temp_dir, f'{n}.pkl'), 'w').close()

            expected_samples = set(combinations(['0', '1', '2'], 2))
            expected_samples.update([(b, a) for a, b in expected_samples])
            actual_samples = set([sample_seg_pair() for _ in range(100)])
            self.assertEqual(actual_samples, expected_samples)

            mark_compared('0', '1', 1)
            expected_samples.remove(('0', '1'))
            expected_samples.remove(('1', '0'))
            actual_samples = set([sample_seg_pair() for _ in range(100)])
            self.assertEqual(actual_samples, expected_samples)

            mark_compared('0', '2', 1)
            expected_samples.remove(('0', '2'))
            expected_samples.remove(('2', '0'))
            actual_samples = set([sample_seg_pair() for _ in range(100)])
            self.assertEqual(actual_samples, expected_samples)

            mark_compared('1', '2', 1)
            with self.assertRaises(IndexError):
                sample_seg_pair()


if __name__ == '__main__':
    unittest.main()
