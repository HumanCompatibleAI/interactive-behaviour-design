import os
import sys

import gym
from gym.utils.play import play

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wrappers.fetch_block_stacking import register, FetchBlockStackingObsWrapper
from wrappers.util_wrappers import RenderObs, SaveObsToInfo, FetchDiscreteActions
from wrappers.wrappers_debug import DrawRewards, DrawObses, DrawStats, GraphRewards


def test_play(env_id):
    print(f"Using env {env_id}")
    env = gym.make(env_id)
    env._max_episode_steps = None
    env._max_episode_seconds = None
    # env = SaveObsToInfo(env)
    # env = DrawRewards(env)
    # env = DrawObses(env, decode_fn=FetchBlockStackingObsWrapper.decode)
    # env = DrawStats(env)
    env = GraphRewards(env, scale=1)
    env.reset()
    env = RenderObs(env)
    env = FetchDiscreteActions(env)

    def callback(prev_obs, obs, action, rew, env_done, info):
        pass

    play(env, callback=callback)


if __name__ == '__main__':
    register()
    test_play('FetchBlockStackingDenseRepeat5BinaryGripper-v0')
