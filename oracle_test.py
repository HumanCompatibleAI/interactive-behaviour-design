import unittest

from mock import patch

from oracle import RateLimiter


class TestLabelRate(unittest.TestCase):

    @patch('time.sleep')
    def test(self, patched_sleep):
        n_steps = 0
        def f():
            return n_steps

        l = RateLimiter(interval_seconds=3.3,
                        decay_rate=True,
                        get_timesteps_fn=f)

        fps = 800
        train_time = 0
        human_time = 0
        n_prefs = 0
        while n_prefs < 5000:
            n_prefs += 1
            human_time += 3.3
            l.sleep()
            time_slept = patched_sleep.call_args_list[-1][0][0]
            train_time += time_slept
            n_steps = train_time * fps
        train_time_hours = train_time / 60 / 60
        human_time_hours = human_time / 60 / 60

        # The DRLHP paper says training took about a day and involved 5 hours of human time
        self.assertAlmostEqual(train_time_hours, 23, places=0)
        self.assertAlmostEqual(human_time_hours, 5, places=0)
