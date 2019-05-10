#!/usr/bin/env python

import argparse
import json
from collections import deque, namedtuple

import random
import time
import traceback

import requests


# TODO: should we be more careful about resolving ties between e.g. 5 segments?
from utils import Timer


class RateLimiter:
    def __init__(self, interval_seconds):
        self.interval = interval_seconds
        self.t = time.time()

    def sleep(self):
        delta = time.time() - self.t
        if delta < self.interval:
            time.sleep(self.interval - delta)
        print("Since last: {:.1f} seconds".format(time.time() - self.t))
        self.t = time.time()


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
    parser.add_argument('min_label_interval_seconds', type=int)
    # '15/45' means 'Alternating periods of 15 minutes giving preferences and 45 minutes of rest'
    parser.add_argument('--schedule')
    args = parser.parse_args()

    if args.schedule:
        work_period_mins, rest_period_mins = map(int, args.schedule.split('/'))
        work_timer = Timer(work_period_mins * 60)
        work_timer.reset()
    else:
        work_period_mins, rest_period_mins = None, None
        work_timer = None

    rate_limiter = RateLimiter(interval_seconds=args.min_label_interval_seconds)

    n = 0
    while True:
        try:
            if args.segment_generation == 'demonstrations':
                demonstrate(args.url)
                n += 1
            elif args.segment_generation == 'drlhp':
                compare(args.url)
                n += 1
            else:
                raise Exception()
        except:
            traceback.print_exc()
            time.sleep(1.0)
        else:
            print(f"Simulated {n} interactions")
        rate_limiter.sleep()

        if work_timer and work_timer.done():
            print("Resting for {} minutes".format(rest_period_mins))
            time.sleep(rest_period_mins * 60)
            work_timer.reset()


if __name__ == '__main__':
    main()
