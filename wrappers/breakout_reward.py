import gym
import numpy as np
from gym import Wrapper, ObservationWrapper, spaces
from gym.envs import register as gym_register

from baselines.common.atari_wrappers import wrap_deepmind, NoopResetEnv, MaxAndSkipEnv

LIFE_LOST_REWARD = -5


class RemoveBreakoutScore(ObservationWrapper):
    def observation(self, observation):
        observation = np.array(observation)
        y = 0
        h = 15
        x = 0
        w = 160
        observation[y:y + h, x:x + w, :] = 0
        return observation


class BreakoutRewardWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.last_ball_lost = None

    def reset(self):
        obs = self.env.reset()
        self.last_ball_lost = False
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        line_just_below_paddle_sum = np.sum(obs[194, :, :])
        if line_just_below_paddle_sum > 5800:
            ball_lost = True
        else:
            ball_lost = False
        if ball_lost and not self.last_ball_lost:
            reward += LIFE_LOST_REWARD
        self.last_ball_lost = ball_lost

        # print(line_just_below_paddle_sum, ball_lost)
        # print(reward)

        return obs, reward, done, info


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


def make_env(dense):
    # The following is equivalent to how Baselines creates an Atari environment.
    # Baselines uses make_atari, which applies NoopResetEnv and MaxAndSkipEnv,
    # then called wrap_deepmind.
    env = gym.make('BreakoutNoFrameskip-v4').unwrapped  # unwrap past TimeLimit
    env = RemoveBreakoutScore(env)
    if dense:
        env = BreakoutRewardWrapper(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    # NB we don't specify scale=True, so observations are not normalized
    env = wrap_deepmind(env, frame_stack=True)

    return env


def register():
    gym_register(
        id='BreakoutDeepMind-v0',
        entry_point=lambda: make_env(dense=False),
        max_episode_steps=4 * 100000,  # frameskip * 100000 - matches Gym original
    )
    gym_register(
        id='BreakoutDeepMindDense-v0',
        entry_point=lambda: make_env(dense=True),
        max_episode_steps=4 * 100000,
    )


def test():
    env = gym.make('BreakoutDeepMindDense-v0')
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
