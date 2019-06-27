import os
import sys

import gym
from gym.utils.play import play

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wrappers.fetch_reach import register as register_env
from wrappers.util_wrappers import SaveObsToInfo, RenderObs, FetchDiscreteActions
from wrappers.wrappers_debug import DrawRewards


def test_play():
    env = gym.make('FetchReach-CustomActionRepeat5ActionLimit0.2-v0')
    env._max_episode_steps = None
    env._max_episode_seconds = None
    env = SaveObsToInfo(env)
    env = DrawRewards(env)
    env = RenderObs(env)
    env = FetchDiscreteActions(env)

    def callback(prev_obs, obs, action, rew, env_done, info):
        pass

    play(env, callback=callback)


if __name__ == '__main__':
    register_env()
    test_play()
