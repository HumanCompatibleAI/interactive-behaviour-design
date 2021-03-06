from enum import Enum
from typing import Union


class RolloutMode(Enum):
    PRIMITIVES = 0
    CUR_POLICY = 1


class RolloutRandomness(Enum):
    SAMPLE_ACTION = 0
    RANDOM_ACTION = 1
    CORRELATED_RANDOM_ACTION = 2


segment_save_mode = None
max_segs = None
render_segments = None
rollout_random_action_prob = None  # type: Union[float, None]
rollout_random_correlation = None  # type: Union[float, None]
rollout_mode = None  # type: Union[RolloutMode, None]
rollout_randomness = None  # type: Union[RolloutRandomness, None]
n_cur_policy = None  # type: Union[int, None]
# ALE is generally safe to use from multiple threads, but we do need to be careful about
# two threads creating environments at the same time:
# https://github.com/mgbellemare/Arcade-Learning-Environment/issues/86
# Any thread which creates environments (which includes restoring from a reset state)
# should acquire this lock before attempting the creation.
env_creation_lock = None
reward_selector = None
n_rl_steps_per_interaction = None
frames_per_segment = None
rollout_noise_sigma = None
reward_predictor_load_polyak_coef = None  # type: Union[float, None]
predicted_reward_normalization = None
log_reward_normalization_every_n_calls = None
predicted_rewards_normalize_mean_std = None
predicted_rewards_normalize_min_max = None
predicted_reward_normalization_norm_loss_coef = None

pids_to_proc_names = {}
