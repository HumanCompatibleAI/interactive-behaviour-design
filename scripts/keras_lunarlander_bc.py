import argparse
import multiprocessing
import os
import sys

import gym
import numpy as np
from gym.envs.box2d.lunar_lander import heuristic
from gym.wrappers import Monitor
from keras import Sequential
from keras.callbacks import TensorBoard, Callback
from keras.engine.saving import load_model
from keras.layers import Dense
from keras.optimizers import Adam

from wrappers.lunar_lander_reward import register

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from wrappers.util_wrappers import LogEpisodeStats


def gen_demonstrations(env_id, log_dir, n_demonstrations):
    register()
    env = gym.make(env_id)
    env = LogEpisodeStats(env, log_dir, '_demo')
    # env = Monitor(env, video_callable=lambda n: True, directory=log_dir, uid=111)

    obses, actions = [], []
    for n in range(n_demonstrations):
        print(f"Generating demonstration {n}...")
        obs, done = env.reset(), False
        while not done:
            action = heuristic(env.unwrapped, obs)
            obses.append(obs)
            actions.append(action)
            obs, reward, done, info = env.step(action)

    return obses, actions


def run_test_env(env_id, log_dir, model_path, model_lock):
    register()
    test_env = gym.make(env_id)
    test_env = LogEpisodeStats(test_env, log_dir, '_test')
    test_env = Monitor(test_env, video_callable=lambda n: n % 7 == 0, directory=log_dir, uid=999)

    n = 0
    while True:
        model_lock.acquire()
        model = load_model(model_path)
        model_lock.release()
        obs, done = test_env.reset(), False
        while not done:
            a = np.argmax(model.predict(np.array([obs]))[0])
            obs, reward, done, info = test_env.step(a)
        sys.stdout.flush()
        sys.stderr.flush()
        print(f"Test episode {n} done!")
        n += 1


class SaveModel(Callback):
    def __init__(self, model_path, model_lock):
        super().__init__()
        self.model_path = model_path
        self.model_lock = model_lock

    def on_epoch_end(self, epoch, logs=None):
        self.model_lock.acquire()
        self.model.save(self.model_path)
        self.model_lock.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()

    print(args.log_dir)

    env_id = 'LunarLanderStatefulStats-v0'

    model = Sequential()
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=4, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-4))

    obses, actions = gen_demonstrations(env_id, os.path.join(args.log_dir, 'demos'), 100)
    obses = np.array(obses)
    actions = np.array(actions)

    ctx = multiprocessing.get_context('spawn')
    model_lock = ctx.Lock()
    model_path = os.path.join(args.log_dir, 'model.h5')
    test_log_dir = os.path.join(args.log_dir, 'test_env')
    test_eps_proc = ctx.Process(target=run_test_env, args=(env_id, test_log_dir, model_path, model_lock))
    model.fit(obses, actions, epochs=1)
    model.save(model_path)
    test_eps_proc.start()

    tb_callback = TensorBoard(args.log_dir)
    model_callback = SaveModel(model_path, model_lock)

    model.fit(obses, actions, epochs=999, callbacks=[tb_callback, model_callback])


if __name__ == '__main__':
    main()
