from enum import Enum


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
n_rl_steps_per_interaction = None
frames_per_segment = None
rollout_noise_sigma = None
goal_sqil = None

pids_to_proc_names = {}
