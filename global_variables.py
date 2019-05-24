segment_save_mode = None
# ALE is generally safe to use from multiple threads, but we do need to be careful about
# two threads creating environments at the same time:
# https://github.com/mgbellemare/Arcade-Learning-Environment/issues/86
# Any thread which creates environments (which includes restoring from a reset state)
# should acquire this lock before attempting the creation.
env_creation_lock = None
reward_selector = None