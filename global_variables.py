from enum import Enum


class RolloutMode(Enum):
    primitives = 0
    cur_policy = 1


class RolloutRandomness(Enum):
    sample_action = 0
    random_action = 1
    correlated_random_action = 2


segment_save_mode = None
max_segs = None
render_segments = None
rollout_random_action_prob = None  # type: float
rollout_random_correlation = None  # type: float
rollout_mode = None  # type: RolloutMode
rollout_randomness = None  # type: RolloutRandomness
n_cur_policy = None  # type: int
# ALE is generally safe to use from multiple threads, but we do need to be careful about
# two threads creating environments at the same time:
# https://github.com/mgbellemare/Arcade-Learning-Environment/issues/86
# Any thread which creates environments (which includes restoring from a reset state)
# should acquire this lock before attempting the creation.
env_creation_lock = None
reward_selector = None
# Let's about for about 5000 interactions in 10 hours
# Assuming 800 steps per second, that's 5000 interactions in 10 * 60 * 60 * 800 = 30M steps
# So one preference ever 5e3/3e7 = 6000 steps
min_n_rl_steps_per_pref = 6e3
