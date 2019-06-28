import argparse
import os
import pickle
import sys

import gym
import pysnooper
from gym.utils.play import play


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import ObsRewardTuple
from wrappers.fetch_reach import register as register_env
from wrappers.util_wrappers import SaveObsToInfo, RenderObs, FetchDiscreteActions
from wrappers.wrappers_debug import DrawRewards



def test_play(save_filename):
    env = gym.make('FetchReach-CustomActionRepeat5ActionLimit0.2-v0')
    env._max_episode_steps = None
    env._max_episode_seconds = None
    env = SaveObsToInfo(env)
    env = DrawRewards(env)
    env = RenderObs(env)
    env = FetchDiscreteActions(env)

    obs_reward_tuples = []

    if save_filename is not None:
        save_file = open(save_filename, 'wb')
    else:
        save_file = None

    def callback(prev_obs, obs, action, rew, env_done, info):
        if save_file is not None:
            obs_reward_tuples.append(ObsRewardTuple(info['obs'], rew))

    play(env, callback=callback)
    pickle.dump(obs_reward_tuples, save_file)
    save_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save')
    args = parser.parse_args()
    register_env()
    test_play(args.save)
