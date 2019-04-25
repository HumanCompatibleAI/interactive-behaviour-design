import sys
from collections import deque

import gym
from gym import Wrapper
from gym.envs import register as gym_register

from baselines import logger
from utils import draw_dict_on_image
from wrappers import lunar_lander_stateful
from wrappers.util_wrappers import SaveEpisodeStats, LogEpisodeStats

lunar_lander_stateful.register()


def has_landed_between_flags(env, observation):
    xPos = observation[0]
    isBetweenFlags = abs(xPos) < 0.1
    return has_landed(env, observation) and isBetweenFlags


def has_landed(env, observation):
    xPos, xVel, yVel, = observation[0], observation[2], observation[3]
    bothLegsOnGround = all([env.unwrapped.legs[i].ground_contact for i in range(2)])
    hasStopped = abs(xVel) < 0.1 and abs(yVel) < 0.1
    return bothLegsOnGround and hasStopped


class LunarLanderStatsWrapper(SaveEpisodeStats):
    def __init__(self, env, window_size=10):
        SaveEpisodeStats.__init__(self, env)
        self.landings = deque(maxlen=window_size)
        self.landings_between_flags = deque(maxlen=window_size)
        self.crashes = deque(maxlen=window_size)
        self.successful_landings = deque(maxlen=window_size)
        self.last_obs = None
        self.debug = False

    def reset(self):
        if self.last_obs is not None:

            landed = has_landed(self.env, self.last_obs)
            landed_between_flags = has_landed_between_flags(self.env, self.last_obs)
            # game_over => spacecraft body in contact with ground => we've crashed
            crashed = self.env.unwrapped.game_over
            landed_successfully = landed_between_flags and not crashed

            self.last_stats['landed'] = float(landed)
            self.last_stats['landed_between_flags'] = float(landed_between_flags)
            self.last_stats['landed_successfully'] = float(landed_successfully)
            self.last_stats['crashed'] = float(crashed)

            self.landings.append(landed)
            self.landings_between_flags.append(landed_between_flags)
            self.crashes.append(crashed)
            self.successful_landings.append(landed_successfully)

            if len(self.landings) == self.landings.maxlen:
                self.last_stats['landing_rate'] = self.landings.count(True) / len(self.landings)
                self.last_stats['landing_between_flags_rate'] = \
                    self.landings_between_flags.count(True) / len(self.landings_between_flags)
                self.last_stats['crash_rate'] = self.crashes.count(True) / len(self.crashes)
                self.last_stats['successful_landing_rate'] = \
                    self.successful_landings.count(True) / len(self.successful_landings)

        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs
        return obs, reward, done, info

    def render(self, mode='human', **kwargs):
        if not self.debug or mode != 'rgb_array':
            return self.env.render(mode, **kwargs)

        im = self.env.render(mode='rgb_array', **kwargs)
        if self.last_obs is not None:
            xPos, xVel, yVel, = self.last_obs[0], self.last_obs[2], self.last_obs[3]
            isBetweenFlags = abs(xPos) < 0.1
            bothLegsOnGround = all([self.env.unwrapped.legs[i].ground_contact for i in range(2)])
            hasStopped = abs(xVel) < 0.1 and + abs(yVel) < 0.1
            landed = isBetweenFlags and bothLegsOnGround and hasStopped
            d = {
                'xPos': '{:.2f}'.format(xPos),
                'xVel': '{:.2f}'.format(xVel),
                'yVel': '{:.2f}'.format(yVel),
                'isBetweenFlags': isBetweenFlags,
                'bothLegsOnGround': bothLegsOnGround,
                'hasStopped': hasStopped,
                'landed': landed,
                'game_over': self.env.unwrapped.game_over
            }
            im = draw_dict_on_image(im, d)
        return im


class LunarLanderEarlyTermination(Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if has_landed(self.env, observation):
            done = True
        return observation, reward, done, info


def make_env(debug):
    env = gym.make('LunarLanderStateful-v0').unwrapped  # unwrap past TimeLimit
    env = LunarLanderStatsWrapper(env)
    env.debug = debug
    return env


def make_stats_env(early_termination=True):
    env_id = 'LunarLanderStateful-v0'
    env = gym.make(env_id).unwrapped
    if early_termination:
        env = LunarLanderEarlyTermination(env)
    env = LunarLanderStatsWrapper(env)
    env = LogEpisodeStats(env, log_dir=logger.get_dir())
    return env


def register(debug=False):
    gym_register(
        id='LunarLanderStatefulStats-v0',
        entry_point=lambda: make_env(debug),
        max_episode_steps=1000,
    )


if __name__ == '__main__':
    register(debug=True)
    sys.argv = ['foo', 'LunarLanderStatefulEarlyTermination-v0']
    from examples.agents import keyboard_agent

    keyboard_agent  # dummy statement so that keyboard_agent import isn't optimized away
