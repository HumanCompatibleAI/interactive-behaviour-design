from enum import Enum


class RolloutMode(Enum):
    primitives = 0
    cur_policy = 1

class RolloutRandomness(Enum):
    sample = 0
    noise = 1


segment_save_mode = None
max_segs = None
render_segments = None
rollout_action_noise = None
rollout_mode = None  # type: RolloutMode
rollout_randomness = None  # type: RolloutRandomness
n_cur_policy = None  # type: int
# ALE is generally safe to use from multiple threads, but we do need to be careful about
# two threads creating environments at the same time:
# https://github.com/mgbellemare/Arcade-Learning-Environment/issues/86
# Any thread which creates environments (which includes restoring from a reset state)
# should acquire this lock before attempting the creation.
env_creation_lock = None
