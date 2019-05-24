import gym
import numpy as np
from gym import Wrapper, ObservationWrapper
from gym.spaces import Discrete, Box

from wrappers.fetch_pick_and_place import RandomInitialPosition
from wrappers.fetch_pick_and_place_register import register as pp_register
from wrappers.wrappers_debug import DrawRewards


class RenderObs(ObservationWrapper):
    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        im = self.env.render(mode='rgb_array')
        self.observation_space = Box(low=0, high=255, shape=im.shape)

    def observation(self, observation):
        return self.env.render(mode='rgb_array')


class FetchPickAndPlaceDiscreteActions(Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env)
        self.action_space = Discrete(9)
        self.closed = False

    def get_action_meanings(self):
        return ['NOOP',
                'LEFT',
                'RIGHT',
                'FORWARD',
                'BACKWARD',
                'UP',
                'DOWN',
                'OPEN',
                'CLOSE', ]

    def get_keys_to_action(self):
        return {
            (ord('w'),): self.get_action_meanings().index('FORWARD'),
            (ord('s'),): self.get_action_meanings().index('BACKWARD'),
            (ord('a'),): self.get_action_meanings().index('LEFT'),
            (ord('d'),): self.get_action_meanings().index('RIGHT'),
            (ord('o'),): self.get_action_meanings().index('UP'),
            (ord('l'),): self.get_action_meanings().index('DOWN'),
            (ord('i'),): self.get_action_meanings().index('OPEN'),
            (ord('k'),): self.get_action_meanings().index('CLOSE'),
        }

    def step(self, action):
        if action == self.get_action_meanings().index('NOOP'):
            caction = [0.0, 0.0, 0.0, 0.0]
        elif action == self.get_action_meanings().index('BACKWARD'):
            caction = [0.1, 0.0, 0.0, 0.0]
        elif action == self.get_action_meanings().index('FORWARD'):
            caction = [-0.1, 0.0, 0.0, 0.0]
        elif action == self.get_action_meanings().index('RIGHT'):
            caction = [0.0, 0.1, 0.0, 0.0]
        elif action == self.get_action_meanings().index('LEFT'):
            caction = [0.0, -0.1, 0.0, 0.0]
        elif action == self.get_action_meanings().index('UP'):
            caction = [0.0, 0.0, 0.1, 0.0]
        elif action == self.get_action_meanings().index('DOWN'):
            caction = [0.0, 0.0, -0.1, 0.0]
        elif action == self.get_action_meanings().index('OPEN'):
            caction = [0.0, 0.0, 0.0, 0.1]
        elif action == self.get_action_meanings().index('CLOSE'):
            caction = [0.0, 0.0, 0.0, -0.1]
        else:
            raise RuntimeError(action)

        return self.env.step(caction)


class SaveObs(Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['obs'] = np.copy(obs)
        return obs, reward, done, info


def test_play(env_id):
    print(f"Using env {env_id}")
    env = gym.make(env_id)
    env._max_episode_steps = None
    env._max_episode_seconds = 1
    env = RandomInitialPosition(env)
    env = SaveObs(env)
    env = DrawRewards(env)
    env = RenderObs(env)
    env = FetchPickAndPlaceDiscreteActions(env)

    def callback(prev_obs, obs, action, rew, env_done, info):
        pass

    from gym.utils.play import play
    play(env, callback=callback)


if __name__ == '__main__':
    pp_register()
    test_play('FetchPickAndPlace-Repeat1-ContGripper-NoGripObs-1InitialBlockPos-FixedGoal-Delta-GripperBonuses-v0')
