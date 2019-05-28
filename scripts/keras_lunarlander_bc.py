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
from gym import Wrapper
from gym.envs.box2d.lunar_lander import heuristic
from gym.wrappers import Monitor
from keras import Sequential
from keras.backend.tensorflow_backend import set_session
from keras.engine.saving import load_model
from keras.layers import Dense
from keras.optimizers import Adam

from lunarlander_manual import Demonstration
from wrappers.lunar_lander_reward import register
from wrappers.util_wrappers import SaveEpisodeStats

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
    env = SaveEpisodeStats(env, log_dir, '_demo')
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
    global human_agent_action
    global human_wants_pause

    register()
    test_env = gym.make(env_id)
    test_env = SaveEpisodeStats(test_env, log_dir, '_test')
    test_env = Monitor(test_env, video_callable=lambda n: n % 10 == 0, directory=log_dir, uid=999)

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

class SetLanderWhite(Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset()
        self.env.unwrapped.lander.color1 = (1.0, 1.0, 1.0)
        self.env.unwrapped.lander.color2 = (1.0, 1.0, 1.0)
        return obs

def run_dagger_env(env_id, log_dir, model_path, model_lock, dagger_queue):
    global human_agent_action
    global human_wants_pause

    register()
    dagger_env = gym.make(env_id)
    dagger_env = SetLanderWhite(dagger_env)
    dagger_env = SaveEpisodeStats(dagger_env, log_dir, '_dagger')
    dagger_env = Monitor(dagger_env, video_callable=lambda n: True, directory=log_dir, uid=100)

    dagger_env.render()
    dagger_env.unwrapped.viewer.window.on_key_press = key_press
    dagger_env.unwrapped.viewer.window.on_key_release = key_release

    while True:
        model_lock.acquire()
        model = load_model(model_path)
        model_lock.release()
        obs, done = dagger_env.reset(), False
        dagger_obses, dagger_actions = [], []
        while not done:
            while human_wants_pause:
                dagger_env.render()
                time.sleep(0.1)
            dagger_obses.append(obs)
            dagger_actions.append(human_agent_action)
            a = np.argmax(model.predict(np.array([obs]))[0])
            obs, reward, done, info = dagger_env.step(a)
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            dagger_queue.put((dagger_obses, dagger_actions), block=False)
        except queue.Full:
            print("WARNING: queue was full")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir')
    parser.add_argument('--demonstrations_pkl')
    parser.add_argument('--n_expert_demonstrations', type=int)
    parser.add_argument('--dagger', action='store_true')
    args = parser.parse_args()
    os.makedirs(args.log_dir)

    if not args.demonstrations_pkl and not args.n_expert_demonstrations and not args.dagger:
        raise argparse.ArgumentError("No demonstrations source")

    env_id = 'LunarLanderStatefulStats-v0'

    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_shape=(8,)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=4, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-4))

    obses = np.zeros((0, 8))
    actions = np.zeros((0,))

    if args.n_expert_demonstrations:
        obses, actions = gen_demonstrations(env_id, os.path.join(args.log_dir, 'demos'), args.n_expert_demonstrations)

    if args.demonstrations_pkl:
        with open(args.demonstrations_pkl, 'rb') as f:
            demonstrations = pickle.load(f)
        obses, actions = [], []
        demonstration: Demonstration
        for demonstration in demonstrations:
            obses.extend(demonstration.observations)
            actions.extend(demonstration.actions)

    obses = np.array(obses)
    actions = np.array(actions)

    ctx = multiprocessing.get_context('spawn')
    model_lock = ctx.Lock()
    model_path = os.path.join(args.log_dir, 'model.h5')
    test_log_dir = os.path.join(args.log_dir, 'test_env')
    dagger_log_dir = os.path.join(args.log_dir, 'dagger_env')
    dagger_queue = ctx.Queue(maxsize=10)
    test_eps_proc = ctx.Process(target=run_test_env,
                                args=(env_id, test_log_dir, model_path, model_lock))
    dagger_eps_proc = ctx.Process(target=run_dagger_env,
                                  args=(env_id, dagger_log_dir, model_path, model_lock, dagger_queue))
    model.save(model_path)
    test_eps_proc.start()
    dagger_eps_proc.start()

    epoch_n = 0
    logger = easy_tf_log.Logger(os.path.join(args.log_dir, 'dataset_size'))
    while True:
        if len(obses) > 0:
            history = model.fit(obses, actions, epochs=(epoch_n + 10), initial_epoch=epoch_n)
            for l in history.history['loss']:
                logger.logkv('loss', l)
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


if __name__ == '__main__':
    main()
