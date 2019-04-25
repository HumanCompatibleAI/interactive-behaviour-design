import argparse
import multiprocessing
import os
import pickle
import queue
import sys
import time

import easy_tf_log
import gym
import numpy as np
import tensorflow as tf
from gym.envs.box2d.lunar_lander import heuristic
from gym.wrappers import Monitor
from keras import Sequential
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard, Callback
from keras.engine.saving import load_model
from keras.layers import Dense
from keras.optimizers import Adam

from lunarlander_manual import Demonstration
from wrappers.lunar_lander_reward import register
from wrappers.util_wrappers import LogEpisodeStats

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

human_agent_action = 0
human_wants_pause = False


def key_press(key, mod):
    global human_agent_action, human_wants_pause
    if key == 32:
        human_wants_pause = not human_wants_pause
        return
    a = int(key - ord('0'))
    if a < 0 or a > 3:
        return
    human_agent_action = a


def key_release(key, mod):
    global human_agent_action
    a = int(key - ord('0'))
    if human_agent_action == a:
        human_agent_action = 0


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


def run_test_env(env_id, log_dir, model_path, model_lock, dagger, dagger_queue: multiprocessing.Queue):
    global human_agent_action
    global human_wants_pause

    register()
    test_env = gym.make(env_id)
    test_env = LogEpisodeStats(test_env, log_dir, '_test')
    render_every_n_episodes = 1 if dagger else 7
    test_env = Monitor(test_env, video_callable=lambda n: n % render_every_n_episodes == 0, directory=log_dir, uid=999)

    test_env.render()
    test_env.unwrapped.viewer.window.on_key_press = key_press
    test_env.unwrapped.viewer.window.on_key_release = key_release

    n = 0
    while True:
        model_lock.acquire()
        model = load_model(model_path)
        model_lock.release()
        obs, done = test_env.reset(), False
        dagger_obses, dagger_actions = [], []
        while not done:
            while human_wants_pause:
                test_env.render()
                time.sleep(0.1)
            dagger_obses.append(obs)
            dagger_actions.append(human_agent_action)
            a = np.argmax(model.predict(np.array([obs]))[0])
            obs, reward, done, info = test_env.step(a)
        sys.stdout.flush()
        sys.stderr.flush()
        if dagger:
            try:
                dagger_queue.put((dagger_obses, dagger_actions), block=False)
            except queue.Full:
                print("WARNING: queue was full")
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


class LogDatasetSize(Callback):
    def __init__(self, obses, actions, log_dir):
        super().__init__()
        self.obses = obses
        self.actions = actions
        self.logger = easy_tf_log.Logger(os.path.join(log_dir, 'dataset_size'))

    def on_epoch_end(self, epoch, logs=None):
        self.logger.logkv('n_demonstration_obses', len(self.obses))
        self.logger.logkv('n_demonstration_actions', len(self.actions))


class UpdateDataset(Callback):
    def __init__(self, obses, actions, dagger_queue: multiprocessing.Queue):
        super().__init__()
        self.obses = obses
        self.actions = actions
        self.queue = dagger_queue

    def on_epoch_end(self, epoch, logs=None):
        while True:
            try:
                obses, actions = self.queue.get(timeout=0.5)
            except queue.Empty:
                break
            else:
                self.obses.extend(obses)
                self.actions.extend(actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir')
    parser.add_argument('--demonstrations_pkl')
    parser.add_argument('--dagger', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.log_dir)

    env_id = 'LunarLanderStatefulStats-v0'

    model = Sequential()
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=4, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-4))

    if args.demonstrations_pkl:
        with open(args.demonstrations_pkl, 'rb') as f:
            demonstrations = pickle.load(f)
        obses, actions = [], []
        demonstration: Demonstration
        for demonstration in demonstrations:
            obses.extend(demonstration.observations)
            actions.extend(demonstration.actions)
    else:
        obses, actions = gen_demonstrations(env_id, os.path.join(args.log_dir, 'demos'), 100)
    obses = np.array(obses)
    actions = np.array(actions)

    ctx = multiprocessing.get_context('spawn')
    model_lock = ctx.Lock()
    model_path = os.path.join(args.log_dir, 'model.h5')
    test_log_dir = os.path.join(args.log_dir, 'test_env')
    dagger_queue = ctx.Queue(maxsize=10)
    test_eps_proc = ctx.Process(target=run_test_env,
                                args=(env_id, test_log_dir, model_path, model_lock, args.dagger, dagger_queue))
    model.fit(obses, actions, epochs=1)
    model.save(model_path)
    test_eps_proc.start()

    epoch_n = 0
    logger = easy_tf_log.Logger(os.path.join(args.log_dir, 'dataset_size'))
    while True:
        history = model.fit(obses, actions, epochs=(epoch_n + 10), initial_epoch=epoch_n)
        epoch_n += 10

        model_lock.acquire()
        model.save(model_path)
        model_lock.release()

        obses_l, actions_l = [], []
        while True:
            try:
                d_obses, d_actions = dagger_queue.get(timeout=0.5)
            except queue.Empty:
                print("Got queue empty")
                break
            else:
                print(f"Received {len(d_obses)} DAgger steps")
                obses_l.extend(d_obses)
                actions_l.extend(d_actions)
        if obses_l:
            obses = np.vstack([obses, obses_l])
            actions = np.hstack([actions, actions_l])

        logger.logkv('n_demonstration_obses', len(obses))
        logger.logkv('n_demonstration_actions', len(actions))
        for l in history.history['loss']:
            logger.logkv('loss', l)


if __name__ == '__main__':
    main()
