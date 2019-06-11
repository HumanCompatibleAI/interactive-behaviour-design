import os
import sys

import gym
from gym.utils.play import play

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wrappers.fetch_pick_and_place import RandomInitialPosition
from wrappers.fetch_pick_and_place_register import register as pp_register
from wrappers.util_wrappers import RenderObs, SaveObsToInfo, FetchDiscreteActions
from wrappers.wrappers_debug import DrawRewards


def test_play(env_id):
    print(f"Using env {env_id}")
    env = gym.make(env_id)
    env._max_episode_steps = None
    env._max_episode_seconds = None
    env = RandomInitialPosition(env)
    env = SaveObsToInfo(env)
    env = DrawRewards(env)
    env = RenderObs(env)
    env = FetchDiscreteActions(env)

    def callback(prev_obs, obs, action, rew, env_done, info):
        pass

    play(env, callback=callback)


if __name__ == '__main__':
    pp_register()
    test_play('FetchPickAndPlace-Repeat1-BinaryGripper-NoGripObs-1InitialBlockPos-FixedGoal-NonDelta-GripperBonuses-v0')
