import gym
from gym.utils.play import play

from fetch_block_stacking_env.env import register
from wrappers.util_wrappers import SaveObsToInfo, RenderObs, FetchDiscreteActions
from wrappers.wrappers_debug import DrawRewards


def test_play():
    env = gym.make('FetchBlockStacking-v0')
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
    register()
    test_play()
