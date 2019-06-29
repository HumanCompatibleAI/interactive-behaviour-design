import glob
import multiprocessing
import os
import sys
import time
import traceback

import easy_tf_log
from cloudpickle import cloudpickle

import global_variables
import throttler
from drlhp.pref_db import PrefDBTestTrain
from drlhp.reward_predictor import RewardPredictor
from global_constants import MAIN_PROCESS_LOAD_REWARD_PREDICTOR_EVERY_N_SECONDS, \
    REWARD_PREDICTOR_TRAINING_PROCESS_SAVE_EVERY_N_SECONDS
from utils import load_cpu_config, LogMilliseconds, Timer

FORCE_SAVE_FNAME = 'reward_predictor_force_save'
FORCE_LOAD_FNAME = 'reward_predictor_force_load'


class NoEventError(Exception):
    pass


class FileBasedEventPipe:
    def __init__(self, pipe_path):
        os.mkfifo(pipe_path)
        self.event_pipe = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)

        self.ack_pipe_path = pipe_path + '_ack'
        os.mkfifo(self.ack_pipe_path)

    def get_event(self):
        try:
            event = os.read(self.event_pipe, 512)
        except BlockingIOError:
            raise NoEventError
        if not event:
            raise NoEventError
        return event

    def send_ack(self, ack_str='ack'):
        # We have to wait until someone is listening before opening the pipe;
        # otherwise pipe open hangs
        ack_pipe = open(self.ack_pipe_path, 'w')
        ack_pipe.write(ack_str)
        ack_pipe.close()

    # Expected to be called from another process
    @staticmethod
    def send_event(pipe_path):
        while not os.path.exists(pipe_path):
            print(f"Waiting for pipe '{pipe_path}' to exist...")
            time.sleep(1)
        pipe = os.open(pipe_path, os.O_WRONLY)
        os.write(pipe, b'event')  # Can be anything
        os.close(pipe)

    # Expected to be called from another process
    @staticmethod
    def wait_for_ack(pipe_path):
        pipe = open(pipe_path + '_ack')
        pipe.read()


def drlhp_load_loop(reward_predictor: RewardPredictor, ckpt_path, log_dir):
    force_load_event_pipe = FileBasedEventPipe(os.path.join(log_dir, FORCE_LOAD_FNAME))
    logger = easy_tf_log.Logger(os.path.join(log_dir, 'drlhp_load_loop'))

    while not glob.glob(ckpt_path + '*'):
        print("Waiting for reward predictor checkpoint...")
        time.sleep(5)

    load_timer = Timer(MAIN_PROCESS_LOAD_REWARD_PREDICTOR_EVERY_N_SECONDS)
    load_timer.reset()

    n_successful_loads = 0
    while True:
        try:
            force_load_event_pipe.get_event()
        except NoEventError:
            force_load = False
        else:
            force_load = True

        if not force_load and not load_timer.done():
            time.sleep(1)
            continue

        load_timer.reset()

        try:
            latest_ckpt_path = reward_predictor.get_latest_checkpoint(ckpt_path)
            if force_load:
                reward_predictor.load(latest_ckpt_path)
            else:
                reward_predictor.load_polyak(latest_ckpt_path,
                                             polyak_coef=global_variables.reward_predictor_load_polyak_coef)
            reward_predictor.save(os.path.join(log_dir, 'checkpoints', 'reward_predictor_loaded.ckpt'))
            n_successful_loads += 1
            logger.logkv('reward_predictor_load_loop/n_loads', n_successful_loads)
        except:
            print("Exception while loading reward predictor checkpoint:")
            traceback.print_exc()

        print("Reward predictor loading thread: loaded checkpoint")

        if force_load:
            force_load_event_pipe.send_ack()


def drlhp_train_loop(make_reward_predictor_fn_cloudpickle,
                     run_training: multiprocessing.Value,
                     pref_db_path,
                     save_ckpt_path,
                     log_dir,
                     gpu_n):
    load_cpu_config(log_dir, 'drlhp_training')
    throttler.init(log_dir, 1)

    make_reward_predictor_fn = cloudpickle.loads(make_reward_predictor_fn_cloudpickle)
    reward_predictor = make_reward_predictor_fn('training', gpu_n)  # type: RewardPredictor
    reward_predictor.save(save_ckpt_path)  # So that the checkpoint load thread is quieted
    pref_db = PrefDBTestTrain()
    logger = easy_tf_log.Logger(os.path.join(log_dir, 'drlhp_train_loop'))

    save_ckpt_timer = Timer(duration_seconds=REWARD_PREDICTOR_TRAINING_PROCESS_SAVE_EVERY_N_SECONDS)
    save_ckpt_timer.reset()
    save_ckpt_event_pipe = FileBasedEventPipe(os.path.join(log_dir, FORCE_SAVE_FNAME))

    while True:
        sys.stdout.flush()
        sys.stderr.flush()
        if run_training.value == 0:
            time.sleep(1.0)
            continue

        # This /should/ be fine because we save the preference database atomically,
        # but just in case, let's be careful
        try:
            with LogMilliseconds('reward_predictor_train_loop/prefs_load_time_ms', logger):
                pref_db.load(pref_db_path, verbose=False)
        except:
            print("Exception while loading preference database:")
            traceback.print_exc()
            time.sleep(1.0)
            continue
        if len(pref_db) == 0:
            print("No preferences yet")
            time.sleep(1.0)
            continue

        with LogMilliseconds('reward_predictor_train_loop/train_time_ms', logger):
            try:
                reward_predictor.train(pref_db.train, pref_db.test, val_interval=20, verbose=False)
            except:
                print("Exception while training reward predictor:")
                traceback.print_exc()
                time.sleep(1.0)
                continue

        try:
            save_ckpt_event_pipe.get_event()
        except NoEventError:
            event = False
        else:
            event = True
        if event or save_ckpt_timer.done():
            with LogMilliseconds('reward_predictor_train_loop/ckpt_save_time_ms', logger):
                reward_predictor.save(save_ckpt_path)
            print("Reward predictor training process: saved checkpoint")
            save_ckpt_timer.reset()
        if event:
            save_ckpt_event_pipe.send_ack()
