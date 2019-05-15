import gym
import numpy as np
from gym import ObservationWrapper, spaces
from gym.envs import register as gym_register

from baselines.common.atari_wrappers import wrap_deepmind, NoopResetEnv, MaxAndSkipEnv


class RemoveEnduroSpeedo(ObservationWrapper):
    def observation(self, observation):
        observation = np.array(observation)
        y = 160
        h = 33
        x = 45
        w = 80
        observation[y:y + h, x:x + w, :] = 0
        return observation


class FlattenObs(ObservationWrapper):
    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(obs_shape[0], obs_shape[1] * obs_shape[2]),
                                            dtype=np.uint8)

    def observation(self, obs):
        obs = np.moveaxis(obs, -1, 0)
        return np.hstack(obs)


def make_env(speedo):
    # The following is equivalent to how Baselines creates an Atari environment.
    # Baselines uses make_atari, which applies NoopResetEnv and MaxAndSkipEnv,
    # then called wrap_deepmind.
    env = gym.make('EnduroNoFrameskip-v4').unwrapped  # unwrap past TimeLimit
    if not speedo:
        env = RemoveEnduroSpeedo(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    # NB we don't specify scale=True, so observations are not normalized
    env = wrap_deepmind(env, frame_stack=True)

    return env


def register():
    gym_register(
        id='EnduroDeepMind-v0',
        entry_point=lambda: make_env(speedo=True),
        max_episode_steps=4 * 100000,  # frameskip * 100000 - matches Gym original
    )
    gym_register(
        id='EnduroDeepMindNoSpeedo-v0',
        entry_point=lambda: make_env(speedo=False),
        max_episode_steps=4 * 100000,
    )


def test():
    env = gym.make('EnduroDeepMindNoSpeedo-v0')
    env = FlattenObs(env)

    def callback(prev_obs, obs, action, rew, env_done, info):
        if rew != 0.0:
            print("Reward:", rew)
        print(env_done)

    from gym.utils.play import play
    play(env, callback=callback, zoom=4, fps=30)


if __name__ == '__main__':
    register()
    test()
