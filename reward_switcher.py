from enum import Enum
import numpy as np

from classifier_collection import ClassifierCollection
from drlhp.reward_predictor import RewardPredictor


class RewardSource(Enum):
    CLASSIFIER = 0
    DRLHP = 1
    ENV = 2
    NONE = 3


class RewardSelector:
    def __init__(self, classifiers: ClassifierCollection, reward_predictor: RewardPredictor):
        self.classifiers = classifiers
        self.reward_predictor = reward_predictor
        self.cur_classifier_name = None
        self.cur_reward_source = RewardSource.NONE

    def set_reward_source(self, reward_source):
        assert isinstance(reward_source, RewardSource)
        self.cur_reward_source = reward_source

    def _rewards_from_reward_predictor(self, obs):
        rewards = self.reward_predictor.raw_rewards(obs)[0]
        assert rewards.shape == (obs.shape[0],)
        return rewards

    def _rewards_from_classifier(self, obs):
        if self.cur_classifier_name is None:
            print("Warning: classifier not set")
            return np.array([0.0] * obs.shape[0])
        probs = self.classifiers.predict_positive_prob(self.cur_classifier_name, obs)
        assert probs.shape == (obs.shape[0],)
        rewards = (probs >= 0.5).astype(np.float32)
        return rewards

    def rewards(self, obses, env_rewards):
        # Expect batches
        obses = np.array(obses)
        env_rewards = np.array(env_rewards)
        assert env_rewards.shape[0] == obses.shape[0]
        assert len(env_rewards.shape) == 1, env_rewards.shape

        if self.cur_reward_source == RewardSource.NONE:
            return np.zeros_like(env_rewards)
        elif self.cur_reward_source == RewardSource.ENV:
            return env_rewards
        elif self.cur_reward_source == RewardSource.CLASSIFIER:
            return self._rewards_from_classifier(obses)
        elif self.cur_reward_source == RewardSource.DRLHP:
            return self._rewards_from_reward_predictor(obses)
        else:
            raise Exception(f"Invalid reward source '{self.cur_reward_source}'")
