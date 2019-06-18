import glob
import multiprocessing
import os
import sys
import time
import traceback

import easy_tf_log
from cloudpickle import cloudpickle

import throttler
from drlhp.pref_db import PrefDBTestTrain
from drlhp.reward_predictor import RewardPredictor
from global_constants import MAIN_PROCESS_LOAD_REWARD_PREDICTOR_EVERY_N_SECONDS, \
    REWARD_PREDICTOR_TRAINING_PROCESS_SAVE_EVERY_N_SECONDS
from utils import load_cpu_config, find_latest_checkpoint, LogMilliseconds, Timer


def drlhp_load_loop(reward_predictor: RewardPredictor, ckpt_path, log_dir):
    logger = easy_tf_log.Logger(os.path.join(log_dir, 'drlhp_load_loop'))

    while not glob.glob(ckpt_path + '*'):
        print("Waiting for reward predictor checkpoint...")
        time.sleep(5)

    n_successful_loads = 0
    while True:
        time.sleep(MAIN_PROCESS_LOAD_REWARD_PREDICTOR_EVERY_N_SECONDS)
        try:
            latest_ckpt_path = find_latest_checkpoint(ckpt_path)
            reward_predictor.load(latest_ckpt_path)
            n_successful_loads += 1
            logger.logkv('reward_predictor_load_loop/n_loads', n_successful_loads)
        except:
            print("Exception while loading reward predictor checkpoint:")
            traceback.print_exc()


def drlhp_train_loop(make_reward_predictor_fn_cloudpickle,
                     run_training: multiprocessing.Value,
                     pref_db_path,
                     save_ckpt_path,
                     log_dir,
                     gpu_n):
    load_cpu_config(log_dir, 'drlhp_training')
    throttler.init(log_dir, 1)

    reward_predictor = cloudpickle.loads(make_reward_predictor_fn_cloudpickle)('training', gpu_n)  # type: RewardPredictor
    reward_predictor.save(save_ckpt_path)  # So that the checkpoint load thread is quieted
    pref_db = PrefDBTestTrain()
    logger = easy_tf_log.Logger(os.path.join(log_dir, 'drlhp_train_loop'))

    ckpt_timer = Timer(duration_seconds=REWARD_PREDICTOR_TRAINING_PROCESS_SAVE_EVERY_N_SECONDS)
    ckpt_timer.reset()

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
                pref_db.load(pref_db_path)
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
                reward_predictor.train(pref_db.train, pref_db.test, val_interval=20)
            except:
                print("Exception while training reward predictor:")
                traceback.print_exc()
                time.sleep(1.0)
                continue

        if ckpt_timer.done():
            with LogMilliseconds('reward_predictor_train_loop/ckpt_save_time_ms', logger):
                reward_predictor.save(save_ckpt_path)
            ckpt_timer.reset()
