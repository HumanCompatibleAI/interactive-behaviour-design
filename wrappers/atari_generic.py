import gym
from gym.wrappers import TimeLimit

from baselines.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, wrap_deepmind


def make_atari_env_with_preprocessing(env_id):
    # The following is equivalent to how Baselines creates an Atari environment.
    # Baselines uses make_atari, which applies NoopResetEnv and MaxAndSkipEnv,
    # then called wrap_deepmind.
    env = gym.make(env_id).unwrapped  # unwrap past TimeLimit
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    # NB we don't specify scale=True, so observations are not normalized
    env = wrap_deepmind(env, frame_stack=True)
    env = TimeLimit(env, max_episode_steps=10000)  # matches Gym default
    return env
