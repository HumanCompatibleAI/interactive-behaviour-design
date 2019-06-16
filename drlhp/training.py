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
from global_constants import SYNC_REWARD_PREDICTOR_EVERY_N_SECONDS
from utils import load_cpu_config, find_latest_checkpoint


def drlhp_load_loop(reward_predictor: RewardPredictor, ckpt_path, log_dir):
    logger = easy_tf_log.Logger(os.path.join(log_dir, 'drlhp_load_loop'))

    while not glob.glob(ckpt_path + '*'):
        print("Waiting for reward predictor checkpoint...")
        time.sleep(5)

    n_successful_loads = 0
    while True:
        time.sleep(SYNC_REWARD_PREDICTOR_EVERY_N_SECONDS)
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

    while True:
        sys.stdout.flush()
        sys.stderr.flush()
        if run_training.value == 0:
            time.sleep(1.0)
            continue

        t1 = time.time()
        # This /should/ be fine because we save the preference database atomically,
        # but just in case, let's be careful
        try:
            pref_db.load(pref_db_path)
        except:
            print("Exception while loading preference database:")
            traceback.print_exc()
            time.sleep(1.0)
            continue
        t2 = time.time()

        if len(pref_db) == 0:
            print("No preferences yet")
            time.sleep(1.0)
            continue

        t3 = time.time()
        try:
            reward_predictor.train(pref_db.train, pref_db.test, val_interval=20)
        except:
            print("Exception while training reward predictor:")
            traceback.print_exc()
            time.sleep(1.0)
            continue
        t4 = time.time()

        load_time_ms = (t2 - t1) * 1000
        train_time_ms = (t4 - t3) * 1000
        ratio = train_time_ms / load_time_ms
        logger.logkv('reward_predictor_train_loop/load_time_ms', load_time_ms)
        logger.logkv('reward_predictor_train_loop/train_time_ms', train_time_ms)
        logger.logkv('reward_predictor_train_loop/train_load_time_ratio', ratio)

        reward_predictor.save(save_ckpt_path)
