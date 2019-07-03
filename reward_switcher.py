import os
from enum import Enum

import easy_tf_log
import numpy as np

import global_variables
from classifier_collection import ClassifierCollection
from drlhp.reward_predictor import RewardPredictor, PredictedRewardNormalization
from utils import LimitedRunningStat


class RewardSource(Enum):
    CLASSIFIER = 0
    DRLHP = 1
    ENV = 2
    NONE = 3


class RewardSelector:
    def __init__(self, classifiers: ClassifierCollection, reward_predictor: RewardPredictor, log_dir):
        self.classifiers = classifiers
        self.reward_predictor = reward_predictor
        self.cur_classifier_name = None
        self.cur_reward_source = RewardSource.NONE
        self.reward_stats_env = LimitedRunningStat(shape=(1, ), len=(300 * 100))  # 100 episodes of Fetch
        self.reward_stats_pred = LimitedRunningStat(shape=(1, ), len=(300 * 100))
        self.n_calls = 0
        self.logger = easy_tf_log.Logger(os.path.join(log_dir, 'reward_selector'))

    def set_reward_source(self, reward_source):
        assert isinstance(reward_source, RewardSource)
        self.cur_reward_source = reward_source

    def _rewards_from_reward_predictor(self, obs):
        if global_variables.predicted_reward_normalization == PredictedRewardNormalization.OFF:
            rewards = self.reward_predictor.unnormalized_rewards(obs)[0]
        else:
            rewards = self.reward_predictor.normalized_rewards(obs)
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
            predicted_rewards = np.zeros_like(env_rewards)
        elif self.cur_reward_source == RewardSource.ENV:
            predicted_rewards = env_rewards
        elif self.cur_reward_source == RewardSource.CLASSIFIER:
            predicted_rewards = self._rewards_from_classifier(obses)
        elif self.cur_reward_source == RewardSource.DRLHP:
            predicted_rewards = self._rewards_from_reward_predictor(obses)
        else:
            raise Exception(f"Invalid reward source '{self.cur_reward_source}'")

        for r in env_rewards:
            self.reward_stats_env.push([r])
        for r in predicted_rewards:
            self.reward_stats_pred.push([r])
        if self.n_calls % 1000 == 0:
            self.logger.logkv('reward_selector/rewards_env_mean', self.reward_stats_env.mean)
            self.logger.logkv('reward_selector/rewards_env_std', self.reward_stats_env.std)
            self.logger.logkv('reward_selector/rewards_pred_mean', self.reward_stats_pred.mean)
            self.logger.logkv('reward_selector/rewards_pred_std', self.reward_stats_pred.std)

        self.n_calls += 1

        return predicted_rewards
