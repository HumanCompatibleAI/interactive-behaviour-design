#!/usr/bin/env python3

import argparse
import datetime
import fnmatch
import multiprocessing
import os
import pickle
import re
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.util import deprecation
from tqdm import tqdm

deprecation._PRINT_DEPRECATION_WARNINGS = False


def read_events_by_directory(dirs):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    events_files_by_directory = {}
    for d in dirs:
        events_files = find_files_matching_pattern('events.out.tfevents*', d)
        events_files_by_directory[d] = events_files

    directories = []
    events_files_flat = []
    for directory, events_files in events_files_by_directory.items():
        for events_file in events_files:
            directories.append(directory)
            events_files_flat.append(events_file)

    events_flat = []
    pool_map = pool.imap(read_events_file, events_files_flat)
    for events in tqdm(pool_map, total=len(events_files_flat),
                       desc='Reading events'):
        events_flat.append(events)

    events_list_by_directory = defaultdict(list)
    for k, v in zip(directories, events_flat):
        events_list_by_directory[k].append(v)
    events_by_directory = {}
    for directory, events_list in events_list_by_directory.items():
        events_by_directory[directory] = merge_events(events_list)

    with open('events_by_directory.pkl', 'wb') as f:
        pickle.dump(events_by_directory, f)

    return events_by_directory


def merge_events(events_list):
    assert isinstance(events_list, list)
    merged_events = {}
    for events in events_list:
        assert isinstance(events, dict)
        merged_events.update(events)
    return merged_events


def find_files_matching_pattern(pattern, path):
    result = []
    for root, dirs, files in os.walk(path, followlinks=True):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def read_events_file(events_filename):
    events = {}
    try:
        for event in tf.train.summary_iterator(events_filename):
            for value in event.summary.value:
                if value.tag not in events:
                    events[value.tag] = []
                events[value.tag].append((event.wall_time, value.simple_value))
    except Exception as e:
        print(f"While reading '{events_filename}':", e)
    return events


def prettify_events(events):
    t0 = events[0][0]
    for i in range(len(events)):
        timestamp, value = events[i]
        time = str(datetime.timedelta(seconds=(timestamp - t0)))
        value = "{:.2f}".format(value)
        events[i] = time, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dirs', nargs='+')
    args = parser.parse_args()
    try:
        with open('events_by_directory.pkl', 'rb') as f:
            events_by_directory = pickle.load(f)
    except FileNotFoundError:
        events_by_directory = read_events_by_directory(args.run_dirs)

    events_by_run_by_seed = defaultdict(lambda: defaultdict(dict))
    for directory, events in events_by_directory.items():
        run = directory
        run = re.sub(r'_156.*', '', run)
        run = re.sub(r'fetchr-(.)-', 'fetchr-*-', run)
        seed = re.match(r'fetchr-(.)-', directory).group(1)
        events_by_run_by_seed[run][seed] = events

    for run in sorted(events_by_run_by_seed.keys()):
        print(run)
        events_by_seed = events_by_run_by_seed[run]
        successes = []
        for seed, events in events_by_seed.items():
            success_partial_rate = events['env_test/success_partial_rate']
            prettify_events(success_partial_rate)
            last = success_partial_rate[-1]
            best = max(success_partial_rate, key=lambda tup: tup[1])
            successes.append(last[1] == '1.00' or last[1] == '0.90' or last[1] == '0.50')
            print(' ', seed, best, last)
        n_successes = successes.count(True)
        n_seeds = len(successes)
        print("{}/{} succeeded ({:.0%})".format(n_successes,
                                                n_seeds,
                                                n_successes / n_seeds))
        print()


if __name__ == '__main__':
    main()
