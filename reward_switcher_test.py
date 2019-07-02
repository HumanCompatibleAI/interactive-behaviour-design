import tempfile
import unittest

import numpy as np

from reward_switcher import RewardSelector, RewardSource


class DummyRewardPredictor():
    def normalized_rewards(self, obs):
        return np.array([0])

    def unnormalized_rewards(self, obses):
        return np.array([0])


class SmokeTests(unittest.TestCase):
    def test(self):
        # def __init__(self, classifiers: ClassifierCollection, reward_predictor: RewardPredictor, log_dir):
        with tempfile.TemporaryDirectory() as tmp_dir:
            classifiers = None
            reward_predictor = DummyRewardPredictor()
            # noinspection PyTypeChecker
            reward_selector = RewardSelector(classifiers,
                                             reward_predictor, tmp_dir)
            2
            reward_selector.set_reward_source(RewardSource.DRLHP)
            for _ in range(2000):
                reward_selector.rewards(obses=np.array([0]),
                                        env_rewards=np.array([0]))


if __name__ == '__main__':
    unittest.main()
