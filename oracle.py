#!/usr/bin/env python

import argparse
import glob
import json
import os
import shutil
import tempfile
import time
import traceback
from collections import deque, namedtuple

import easy_tf_log
import requests
import tensorflow as tf

from utils import Timer


class NoMasterPolicyError(Exception):
    pass


class NoRolloutsError(Exception):
    pass


class NoSegmentsError(Exception):
    pass


class RateLimiter:
    def __init__(self, interval_seconds, decay_rate, get_timesteps_fn):
        self.initial_rate = 1 / interval_seconds
        self.decay_rate = decay_rate
        self.get_timesteps_fn = get_timesteps_fn
        self.t = time.time()

    def sleep(self):
        if self.decay_rate:
            try:
                n_steps = self.get_timesteps_fn()
            except NoMasterPolicyError:
                # We're pretraining
                rate = self.initial_rate
            else:
                # From DRLHP paper
                rate = self.initial_rate * 5e6 / (n_steps + 5e6)
                print("At timestep {:.1e}, label rate is {:.2f}".format(n_steps, rate))
        else:
            rate = self.initial_rate
        interval = 1 / rate

        delta = time.time() - self.t
        if delta < interval:
            time.sleep(interval - delta)
        self.t = time.time()


def get_n_training_timesteps(log_dir):
    event_files = glob.glob(os.path.join(log_dir, 'policy_master', 'events*'))
    if not event_files:
        raise NoMasterPolicyError

    assert len(event_files) == 1
    events_file = event_files[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Is it going to be a problem if we're reading while the training process is writing?
        # I'm not sure, but let's be careful
        shutil.copy(events_file, temp_dir)
        n_steps = 0
        try:
            for event in tf.train.summary_iterator(os.path.join(temp_dir, os.path.basename(events_file))):
                for value in event.summary.value:
                    if value.tag == 'policy_master/n_total_steps':
                        n_steps = max(n_steps, value.simple_value)
        except Exception as e:
            raise Exception(f"While reading '{events_file}':" + str(e))

    return n_steps


def choose_best_segment(segment_dict):
    """
    Returns segment with highest reward.
    If all segments have same reward, return None.
    """
    T = namedtuple('HashNameRewardTuple', ['hash', 'name', 'reward'])
    hash_name_reward_tuples = [T(seg_hash, policy_name, sum(rewards))
                               for seg_hash, (policy_name, vid_filename, rewards) in segment_dict.items()]
    print("Segments:", hash_name_reward_tuples)

    rewards = [t.reward for t in hash_name_reward_tuples]
    if len(set(rewards)) == 1:
        best_hash = best_policy_name = None
    else:
        best_hash, best_policy_name, _ = sorted(hash_name_reward_tuples, key=lambda t: t.reward)[-1]

    return best_hash, best_policy_name


def compare(url):
    response = requests.get(url + '/get_comparison')
    response.raise_for_status()
    if response.text == 'No segments available':
        raise NoRolloutsError
    segment_dict = response.json()
    if not segment_dict:
        raise Exception("Empty segment dictionary")
    best_hash, _ = choose_best_segment(segment_dict)
    hashes = list(segment_dict.keys())
    if best_hash is None:
        pref = [0.5, 0.5]
        hash1, hash2 = hashes
    else:
        pref = [1.0, 0.0]
        hash1 = best_hash
        if best_hash == hashes[0]:
            hash2 = hashes[1]
        else:
            hash2 = hashes[0]
    print("Sending preference:", pref, hash1, hash2)
    d = {'hash1': hash1, 'hash2': hash2, 'pref': json.dumps(pref)}
    requests.post(url + '/prefer_segment', data=d).raise_for_status()


chosen_policy_names = deque(maxlen=5)


def choose_segment_for_demonstration(segment_dict):
    """
    If all segments have same reward, return a random segment.
    Also, detect too many redos.
    """
    best_hash, best_policy_name = choose_best_segment(segment_dict)

    if best_hash is None:
        return None, None

    chosen_policy_names.append(best_policy_name)

    # If we've chosen 'redo' too many times, try something else
    if len(chosen_policy_names) == chosen_policy_names.maxlen and all([p == 'redo' for p in chosen_policy_names]):
        print("Chosen 'redo' too many times; choosing something else")
        for hash, (policy_name, vid_filename, reward) in segment_dict.items():
            if policy_name == 'redo':
                del segment_dict[hash]
                break
        chosen_policy_names.clear()
        return choose_segment_for_demonstration(segment_dict)

    return best_hash, best_policy_name


def demonstrate(url):
    response = requests.get(url + '/get_rollouts')
    response.raise_for_status()
    if response.text == 'No rollouts available':
        raise NoRolloutsError

    print(response.json())  # TODO debugging, deleteme
    group_name, demonstrations_dict = response.json()
    best_hash, best_policy_name = choose_segment_for_demonstration(demonstrations_dict)
    if best_hash is None:
        best_hash = 'equal'
    print(f"Choosing {best_hash} ({best_policy_name})")
    request_url = url + f'/choose_rollout?group={group_name}&hash={best_hash}&policies='
    requests.get(request_url).raise_for_status()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('segment_generation', choices=['demonstrations', 'drlhp'])
    parser.add_argument('--seconds_per_label', type=int)
    parser.add_argument('--decay_label_rate', action='store_true')
    parser.add_argument('--schedule')  # '15/45' -> 'Alternating 15 minutes giving preferences/45 minutes of rest'
    parser.add_argument('--log_dir')
    args = parser.parse_args()

    if args.decay_label_rate and not args.log_dir:
        raise argparse.ArgumentError("For --decay_label_rate, you must specify --log_dir")

    if args.schedule:
        work_mins, rest_mins = map(int, args.schedule.split('/'))
    else:
        work_mins, rest_mins = float('inf'), None
    work_timer = Timer(work_mins * 60)
    work_timer.reset()

    rate_limiter = RateLimiter(interval_seconds=args.seconds_per_label, decay_rate=args.decay_label_rate,
                               get_timesteps_fn=lambda: get_n_training_timesteps(args.log_dir))

    logger = easy_tf_log.Logger(os.path.join(args.log_dir, 'oracle'))

    n = 0
    last_interaction_time = None
    while True:
        while not work_timer.done():
            try:
                if args.segment_generation == 'demonstrations':
                    demonstrate(args.url)
                elif args.segment_generation == 'drlhp':
                    compare(args.url)
            except (NoRolloutsError, NoSegmentsError):
                time.sleep(1.0)
                continue
            except:
                traceback.print_exc()
                time.sleep(1.0)
                continue
            else:
                n += 1
                print(f"Simulated {n} interactions")
                if last_interaction_time is not None:
                    t_since_last = time.time() - last_interaction_time
                    print("{:.1f} seconds since last interaction".format(t_since_last))
                    logger.logkv('oracle/label_interval', t_since_last)
                    logger.logkv('oracle/label_rate', 1 / t_since_last)
                last_interaction_time = time.time()
                rate_limiter.sleep()

        last_interaction_time = None
        print("Resting for {} minutes".format(rest_mins))
        time.sleep(rest_mins * 60)
        work_timer.reset()


if __name__ == '__main__':
    main()
