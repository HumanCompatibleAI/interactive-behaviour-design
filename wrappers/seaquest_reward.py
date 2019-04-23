from collections import deque

import gym
import numpy as np
from gym import Wrapper, ObservationWrapper, spaces
from gym.envs import register as gym_register

from baselines.common.atari_wrappers import wrap_deepmind, NoopResetEnv, MaxAndSkipEnv, ClipRewardEnv
from utils import draw_dict_on_image
from wrappers.util_wrappers import SeaquestStatsWrapper

DEATH_BRIGHTNESS_THRESHOLD = 220
# We assume we're operating after wrap_deepmind which has clipped the rewards.
# We're getting +1 reward for shooting a submarine or a shark.
# So mildly prioritise picking up a diver over shooting something,
# and making dying worth as much as shooting ten things.
# (Does it matter that these rewards aren't scaled? No! We're not using them for training!
# We're only using them to allow the oracle to choose good segments.)
DIVER_REWARD = 2
GO_UP_WHEN_LOW_OXYGEN_REWARD = 3
DEATH_REWARD = -50
UP_EARLY_REWARD = -51  # To differentiate from DEATH_REWARD


class SeaquestRewardWrapper(Wrapper):

    def __init__(self, env):
        if not all([w in repr(env) for w in ['ClipRewardEnv', 'WarpFrame', 'EpisodicLifeEnv', 'MaxAndSkipEnv']]):
            raise RuntimeError("SeaquestRewardWrapper should be applied after wrap_deepmind")
        Wrapper.__init__(self, env)
        self.obs_max_history = deque(maxlen=10)
        self.last_state = None
        self.last_oxygen = np.float('inf')
        self.low_oxygen = None
        self.episode_start = None
        self.up_actions = [i
                           for i, a in enumerate(self.env.unwrapped.get_action_meanings())
                           if 'UP' in a]
        self.down_actions = [i
                             for i, a in enumerate(self.env.unwrapped.get_action_meanings())
                             if 'DOWN' in a]
        self.debug_dict = {}

    def reset(self):
        obs = self.env.reset()
        self.last_state = self.env.unwrapped.ale.getRAM()
        self.low_oxygen = False
        self.episode_start = True
        return obs

    def detect_death(self, obs):
        if (self.obs_max_history and
                max(self.obs_max_history) < DEATH_BRIGHTNESS_THRESHOLD and
                np.max(obs) > DEATH_BRIGHTNESS_THRESHOLD):
            return True
        else:
            return False

    def max_oxygen(self, obs):
        # If oxygen bar is fully replenished
        # (This pixel is at bottom right of oxygen bar)
        return obs[69][57][-1] == 214

    def set_low_oxygen(self, obs):
        if self.max_oxygen(obs):
            self.low_oxygen = False
        # If oxygen bar has started flashing black/white
        # (This pixel is at the bottom left corner of the oxygen bar)
        if obs[69][26][-1] == 0:
            self.low_oxygen = True

    def detect_diver_pickup(self, state):
        if state[62] > self.last_state[62]:
            return True
        else:
            return False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self.max_oxygen(obs):
            self.episode_start = False

        self.set_low_oxygen(obs)

        oxygen = np.sum(obs[69, :54])  # Oxygen bar row; oxygen is bright pixels; sum to before low oxygen deactivated
        self.last_oxygen = oxygen

        just_died = self.detect_death(obs)
        self.obs_max_history.append(np.max(obs))

        state = self.env.unwrapped.ale.getRAM()
        picked_up_diver = self.detect_diver_pickup(state)
        self.last_state = state

        if just_died:
            # Oxygen bar stops flashing once dead
            self.low_oxygen = False

        if not self.episode_start:
            reward += DEATH_REWARD * just_died
            reward += DIVER_REWARD * picked_up_diver
            if self.low_oxygen:
                if action in self.down_actions:
                    reward -= GO_UP_WHEN_LOW_OXYGEN_REWARD
                elif action in self.up_actions:
                    reward += GO_UP_WHEN_LOW_OXYGEN_REWARD
            if oxygen > self.last_oxygen and not self.low_oxygen:
                reward += UP_EARLY_REWARD

        self.debug_dict['just_died'] = just_died
        self.debug_dict['low_oxygen'] = self.low_oxygen
        self.debug_dict['oxygen'] = oxygen
        self.debug_dict['picked_up_diver'] = picked_up_diver
        self.debug_dict['episode_start'] = self.episode_start

        return obs, reward, done, info

    def render(self, mode='human', **kwargs):
        im = self.env.render(mode='rgb_array')
        im = draw_dict_on_image(im, self.debug_dict, mode='concat')
        return im


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


def make_env(dense, clipped=False):
    env = gym.make('SeaquestNoFrameskip-v4').unwrapped  # unwrap past TimeLimit
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    # NB we don't specify scale=True, so observations are not normalized
    env = wrap_deepmind(env, frame_stack=True)
    env = SeaquestStatsWrapper(env)
    if dense:
        env = SeaquestRewardWrapper(env)
    if clipped:
        # This is also applied by wrap_deepmind
        # Only specify clipped=True if you want to clip the /dense/ reward
        env = ClipRewardEnv(env)
    return env


def register():
    gym_register(
        id='SeaquestDeepMind-v0',
        entry_point=lambda: make_env(dense=False),
        max_episode_steps=4 * 100000,  # frameskip * 100000 - matches Gym original
    )
    gym_register(
        id='SeaquestDeepMindDense-v0',
        entry_point=lambda: make_env(dense=True),
        max_episode_steps=4 * 100000,
    )
    gym_register(
        id='SeaquestDeepMindDenseClipped-v0',
        entry_point=lambda: make_env(dense=True, clipped=True),
        max_episode_steps=4 * 100000,
    )


def test():
    env = gym.make('SeaquestDeepMindDense-v0')
    env = FlattenObs(env)

    def callback(prev_obs, obs, action, rew, env_done, info):
        if rew != 0.0:
            print("Reward:", rew)

    from gym.utils.play import play
    play(env, callback=callback, zoom=4, fps=30)


if __name__ == '__main__':
    register()
    test()
