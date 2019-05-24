import argparse
import multiprocessing
import os
import sys

import gym
import numpy as np
import tensorflow as tf
from gym.wrappers import Monitor, FlattenDictWrapper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wrappers.wrappers_debug import DrawActions, DrawRewards
from wrappers.fetch_pick_and_place import FetchStatsWrapper, FetchPickAndPlaceObsWrapper
from wrappers.fetch_pick_and_place_register import register
from wrappers.util_wrappers import LogEpisodeStats


class Oracle:
    def __init__(self):
        self.gripping = None
        self.step_n = None
        self.last_action = None

    def reset(self):
        self.gripping = False
        self.step_n = 0

    def get_action_smooth(self, obs):
        gripper_to_block = obs[3:6] - obs[:3]
        block_to_target = obs[25:28] - obs[3:6]
        gripper_width = np.sum(obs[9:11])
        if np.linalg.norm(gripper_to_block) > 0.01 and not self.gripping:
            if np.linalg.norm(gripper_to_block[:2]) > 0.05:
                z = 0
            else:
                z = gripper_to_block[2]
            action = np.concatenate([3 * gripper_to_block[:2], [3 * z, 1]])
        elif gripper_width > 0.05:
            action = np.array([0, 0, 0, -1])
        else:
            self.gripping = True
            action = np.concatenate([3 * block_to_target, [-1]])
        return action

    def get_action_jerky(self, obs):
        if self.step_n % 10 != 0:
            self.step_n += 1
            return self.last_action

        gripper_to_block = obs[:3]
        block_to_target = obs[3:6]
        gripper_width = obs[6]
        if np.linalg.norm(gripper_to_block) > 0.02 and not self.gripping:
            di = np.argmax(np.abs(gripper_to_block))
            action = np.array([0., 0., 0., 1.])
            action[di] = 0.1 * np.sign(gripper_to_block[di])
        elif gripper_width > 0.055:
            action = [0., 0., 0., -0.002]
        else:
            self.gripping = True
            di = np.argmax(np.abs(block_to_target))
            action = np.array([0., 0., 0., -1.])
            action[di] = 0.1 * np.sign(block_to_target[di])

        self.last_action = action
        self.step_n += 1
        return action


def gen_demonstrations(env_id, log_dir, n_demonstrations, seed):
    register()
    env = gym.make(env_id)
    env.seed(seed)
    env._max_episode_steps = 500
    env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env = FetchStatsWrapper(env)
    env = FetchPickAndPlaceObsWrapper(env, include_grip_obs=True)
    env = LogEpisodeStats(env, log_dir, '_demo')

    # from wrappers.wrappers_debug import DrawObses
    # env = DrawObses(env)
    # env = DrawActions(env)
    # env = Monitor(env, video_callable=lambda n: True, directory=log_dir, uid=111)

    obses, actions = [], []
    oracle = Oracle()
    for n in range(n_demonstrations):
        print(f"Generating demonstration {n}...")
        obs, done = env.reset(), False
        oracle.reset()
        while not done:
            action = oracle.get_action_jerky(obs)
            obses.append(obs[:6])
            actions.append(action)
            obs, reward, done, info = env.step(action)

    return obses, actions


def loss(y_true, y_pred):
    # TODO check
    from keras import backend as K
    return K.mean(K.sum(K.square(y_true - y_pred), axis=-1), axis=0)


def run_test_env(env_id, log_dir, model_path, model_lock, seed):
    from keras.engine.saving import load_model
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    register()
    test_env = gym.make(env_id)
    test_env.seed(seed)
    test_env._max_episode_steps = 250
    test_env = FlattenDictWrapper(test_env, ['observation', 'desired_goal'])
    test_env = FetchStatsWrapper(test_env)
    test_env = FetchPickAndPlaceObsWrapper(test_env, include_grip_obs=False)
    test_env = DrawActions(test_env)
    test_env = DrawRewards(test_env)
    test_env = LogEpisodeStats(test_env, log_dir, '_test')
    test_env = Monitor(test_env, video_callable=lambda n: n % 7 == 0, directory=log_dir, uid=999)

    n = 0
    while True:
        model_lock.acquire()
        model = load_model(model_path, custom_objects={'loss': loss})
        model_lock.release()
        obs, done = test_env.reset(), False
        while not done:
            a = model.predict(np.array([obs]))[0]
            obs, reward, done, info = test_env.step(a)
        sys.stdout.flush()
        sys.stderr.flush()
        print(f"Test episode {n} done!")
        n += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    from keras import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    model = Sequential()
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=4, activation='tanh'))
    model.compile(loss=loss, optimizer=Adam(lr=1e-3))

    env_id = 'FetchPickAndPlace-v1'
    obses, actions = gen_demonstrations(env_id, os.path.join(args.log_dir, 'demos'), 100, args.seed)
    obses = np.array(obses)
    actions = np.array(actions)

    ctx = multiprocessing.get_context('spawn')
    model_lock = ctx.Lock()
    model_path = os.path.join(args.log_dir, 'model.h5')
    test_log_dir = os.path.join(args.log_dir, 'test_env')
    test_eps_proc = ctx.Process(target=run_test_env, args=(env_id, test_log_dir, model_path, model_lock, args.seed))
    model.fit(obses, actions, epochs=1)
    model.save(model_path)
    test_eps_proc.start()

    while True:
        model.fit(obses, actions, epochs=10)
        model_lock.acquire()
        model.save(model_path)
        model_lock.release()


if __name__ == '__main__':
    main()
