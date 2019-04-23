#!/usr/bin/env python3

"""
Plot metrics by time and by step, plotting a solid line showing the mean of all seeds and a shaded region
showing one standard error between seeds.

# Run me with --runs_dir pointing to a directory that looks like:
#   fetch-0-drlhp_foobar
# where 'fetch' is the environment name, '0' is the seed, 'drlhp' is the run type, and foobar is ignored
"""

# TODOs; error regions too smoothed; lines sometimes start low then increase?; negative success rate

import argparse
import fnmatch
import glob
import multiprocessing
import os
import re
import sys
import unittest
from collections import namedtuple, defaultdict
from functools import partial

import matplotlib
import numpy as np
import scipy.stats
import tensorflow as tf
from matplotlib.pyplot import close, fill_between

matplotlib.use('Agg')

from pylab import plot, xlabel, ylabel, figure, legend, savefig, grid, ylim, xlim


# Event-reading utils

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


def read_all_events(directory):
    events_files = find_files_matching_pattern('events.out.tfevents*', directory)
    pool = multiprocessing.Pool(processes=32)
    events_in_each_file = pool.map(read_events_file, events_files)
    all_events = {}
    for events in events_in_each_file:
        all_events.update(events)
    return all_events


# Value interpolation helper functions

def interpolate_values(x_y_tuples, new_xs):
    xs, ys = zip(*x_y_tuples)
    if new_xs[-1] < xs[0]:
        raise Exception("New x values end before old x values begin")
    if new_xs[0] > xs[-1]:
        raise Exception("New x values start after old x values end")

    new_ys = np.interp(new_xs, xs, ys,
                       left=np.nan, right=np.nan)  # use NaN if we don't have data
    return new_ys


class TestInterpolateValues(unittest.TestCase):
    def test(self):
        timestamps = [0, 1, 2, 3]
        values = [0, 10, 20, 30]
        timestamps2 = [-1, 0, 0.5, 1, 1.1, 3, 3.1]
        interpolated_values = interpolate_values(list(zip(timestamps, values)), timestamps2)
        np.testing.assert_almost_equal(interpolated_values, [np.nan, 0.0, 5.0, 10.0, 11.0, 30.0, np.nan])


# Plotting helper functions

def smooth_values(values, smoothing):
    smoothed_values = []
    last = values[0]
    for v in values:
        smoothed_values.append(smoothing * last + (1 - smoothing) * v)
        last = smoothed_values[-1]
    return smoothed_values


def interpolate_steps(timestamp_value_tuples, timestamp_step_tuples):
    step_timestamps, steps = zip(*timestamp_step_tuples)
    value_timestamps, values = zip(*timestamp_value_tuples)

    if len(timestamp_step_tuples) < len(timestamp_value_tuples):
        # Use step timestamps for interpolation
        steps = steps
        values = interpolate_values(timestamp_value_tuples, step_timestamps)
    elif len(timestamp_value_tuples) < len(timestamp_step_tuples):
        # Use value timestamps for interpolation
        steps = interpolate_values(timestamp_step_tuples, value_timestamps)
        values = values
    else:
        pass

    # interpolate_values uses NaN to signal "couldn't interpolate this value"
    # (because we didn't have data at the beginning or end); let's remove those points
    drop_idxs = []
    for i in range(len(steps)):
        if np.isnan(steps[i]):
            drop_idxs.append(i)
    steps = [steps[i] for i in range(len(steps)) if i not in drop_idxs]
    values = [values[i] for i in range(len(values)) if i not in drop_idxs]

    return steps, values


def find_training_start(run_dir):
    # The auto train script exits once training has started properly
    return os.path.getmtime(os.path.join(run_dir, 'auto_train.log'))


M = namedtuple('M', 'tag name smoothing fillsmoothing')


def detect_metrics(env_name, train_env_key):
    metrics = []
    if env_name == 'Lunar Lander':
        metrics.append(M(f'{train_env_key}/reward_sum', 'Episode reward', 0.95, 0.95))
        metrics.append(M(f'{train_env_key}/crash_rate', 'Crash rate', 0.95, 0.95))
        metrics.append(M(f'{train_env_key}/successful_landing_rate', 'Successful landing rate', 0.95, 0.95))
    if env_name == 'Seaquest':
        metrics.append(M(f'{train_env_key}/reward_sum', 'Episode reward', 0.9, 0.95))
        metrics.append(M(f'{train_env_key}/n_diver_pickups', 'Diver pickups per episode', 0.99, None))
    if env_name == 'Fetch':
        metrics.append(M(f'{train_env_key}/reward_sum_post_wrappers', 'Episode reward', 0.95, 0.95))
        metrics.append(M(f'{train_env_key}/gripper_to_block_cumulative_distance', 'Distance from gripper to block', 0.95, 0.95))
        metrics.append(M(f'{train_env_key}/block_to_target_cumulative_distance', 'Distance from block to target', 0.95, 0.95))
        metrics.append(M(f'{train_env_key}/block_to_target_min_distance', 'Minimum distance from block to target', 0.95, 0.95))
        metrics.append(M(f'{train_env_key}/ep_frac_aligned_with_block', 'Fraction of episode aligned with block', 0.95, 0.95))
        metrics.append(M(f'{train_env_key}/ep_frac_gripping_block', 'Fraction of episode gripping block', 0.95, 0.95))
        metrics.append(M(f'{train_env_key}/success_rate', 'Success rate', 0.95, 0.95))
    return metrics


def make_timestamps_relative_hours(events):
    for timestamp_value_tuples in events.values():
        first_timestamp = timestamp_value_tuples[0][0]
        for n, (timestamp, value) in enumerate(timestamp_value_tuples):
            timestamp_value_tuples[n] = ((timestamp - first_timestamp) / 3600, value)


def downsample(xs, ys, n_samples):
    """
    Downsample by dividing xs into n_samples equally-sized ranges,
    then calculating the mean of ys in each range.

    If there aren't ys in some of the ranges, interpolate.
    """
    bin_means, bin_edges, _ = scipy.stats.binned_statistic(x=xs, values=ys, statistic='mean',
                                                           bins=n_samples  # no. of equal-width bins
                                                           )

    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[1:] - bin_width / 2

    non_empty_bins = ~np.isnan(bin_means)
    nonempty_bin_means = bin_means[non_empty_bins]
    nonempty_bin_centers = bin_centers[non_empty_bins]

    interped_bin_means = np.interp(bin_centers, nonempty_bin_centers, nonempty_bin_means)

    return bin_centers, interped_bin_means


class TestDownsample(unittest.TestCase):
    def test_downsample(self):
        xs = np.arange(13)
        ys = 2 * np.arange(13)
        xs_downsampled, ys_downsampled = downsample(xs, ys, n_samples=3)
        # Bin 1: [0, 4), center = 2
        # Bin 2: [4, 8), center = 6
        # Bin 3: [8, 12], center = 10
        np.testing.assert_array_equal(xs_downsampled, [2, 6, 10])
        # Bin 1: [0, 8), mean = 3.0
        # Bin 2: [8, 16), mean = 11.0
        # Bin 3: [16, 24], mean = 20
        np.testing.assert_array_equal(ys_downsampled, [3, 11, 20])

    def test_downsample_missing_data(self):
        xs = np.arange(13)
        ys = 2 * np.arange(13)
        xs = np.concatenate([xs[:4], xs[8:]])
        ys = np.concatenate([ys[:4], ys[8:]])
        # Bin 2 has no values in; should have been interpolated
        xs_downsampled, ys_downsampled = downsample(xs, ys, n_samples=3)
        np.testing.assert_array_equal(xs_downsampled, [2, 6, 10])
        np.testing.assert_array_equal(ys_downsampled, [3, 11.5, 20])


def plot_averaged(xs_list, ys_list, smoothing, fillsmoothing, color, label):
    # Interpolate all data to have common x values
    all_xs = set([x for xs in xs_list for x in xs])
    all_xs = sorted(list(all_xs))
    for n in range(len(xs_list)):
        ys_list[n] = interpolate_values(x_y_tuples=list(zip(xs_list[n], ys_list[n])),
                                        new_xs=all_xs)
    # interpolate_values uses NaN to signal "couldn't interpolate this value"
    # (because we didn't have data at the beginning or end); let's remove those points
    drop_idxs = []
    for ys in ys_list:
        for i in range(len(ys)):
            if np.isnan(ys[i]):
                drop_idxs.append(i)
    all_xs = [all_xs[i] for i in range(len(all_xs)) if i not in drop_idxs]
    for n in range(len(ys_list)):
        ys_list[n] = [ys_list[n][i] for i in range(len(ys_list[n])) if i not in drop_idxs]
    assert all([len(ys) == len(all_xs) for ys in ys_list])

    # Downsample /before/ smoothing so that we get the same level of smoothness no matter how dense the data
    plot_width_pixels = 1500  # determined by manually checking the figure
    xs_downsampled = None
    ys_downsampled_list = []
    for ys in ys_list:
        xs_downsampled, ys_downsampled = downsample(all_xs, ys, plot_width_pixels)
        assert len(ys_downsampled) == len(xs_downsampled)
        ys_downsampled_list.append(ys_downsampled)

    mean_ys = np.mean(ys_downsampled_list, axis=0)  # Average across seeds
    smoothed_mean_ys = smooth_values(mean_ys, smoothing=smoothing)
    plot(xs_downsampled, smoothed_mean_ys, color=color, label=label, alpha=0.9)

    if fillsmoothing is not None:
        std = np.std(ys_downsampled_list, axis=0)
        lower = smooth_values(smoothed_mean_ys - std, fillsmoothing)
        upper = smooth_values(smoothed_mean_ys + std, fillsmoothing)
        fill_between(xs_downsampled, lower, upper, color=color, alpha=0.2)
        min_val, max_val = np.min(lower), np.max(upper)
    else:
        min_val, max_val = np.min(smoothed_mean_ys), np.max(smoothed_mean_ys)

    grid(True)

    return min_val, max_val


def parse_run_name(run_dir):
    match = re.search(r'([^-]*)-([\d])-([^_]*)_', run_dir)  # e.g. fetch-0-drlhp_foobar
    if match is None:
        raise Exception(f"Couldn't parse run name '{run_dir}'")
    env_shortname = match.group(1)
    seed = match.group(2)
    run_type = match.group(3)

    env_shortname_to_env_name = {
        'fetchpp': 'Fetch',
        'fetch': 'Fetch',
        'lunarlander': 'Lunar Lander',
        'seaquest': 'Seaquest'
    }
    env_name = env_shortname_to_env_name[env_shortname]
    run_type = run_type.upper()

    return env_name, run_type, seed


def filter_pretraining_events(run_dir, events):
    training_start_timestamp = find_training_start(run_dir)
    for tag in events:
        events[tag] = [(t, v) for t, v in events[tag] if t >= training_start_timestamp]
        if not events[tag]:
            del events[tag]
    # Reset the steps to start from 0 after the pretraining period
    first_step = events['policy_master/n_total_steps'][0][1]
    events['policy_master/n_total_steps'] = [(t, step - first_step)
                                             for t, step in events['policy_master/n_total_steps']]


def get_values_by_step(events, metric, max_steps):
    steps, values = interpolate_steps(events[metric.tag], events['policy_master/n_total_steps'])
    if max_steps:
        values = np.extract(np.array(steps) < max_steps, values)
        steps = np.extract(np.array(steps) < max_steps, steps)
    return steps, values


def get_values_by_time(events, metric, max_hours):
    timestamps, values = zip(*events[metric.tag])
    if max_hours:
        values = np.extract(np.array(timestamps) < max_hours, values)
        timestamps = np.extract(np.array(timestamps) < max_hours, timestamps)
    return timestamps, values


def add_steps_to_bc_run(events_by_env_name_by_run_type_by_seed, max_steps):
    # BC runs don't actually interact with the environment, so don't log n_total_steps.
    # But we still want BC results to appear on the graphs by steps.
    # So let's fake the step values.
    #
    # We want to create step values so that we effectively take the first section of the BC metrics and stretch it to
    # fill the space up to max_steps. We shouldn't take /all/ the data, because the other runs might run for longer
    # than max_steps; there we might be taking 3/4 or 4/5 of the data, so if we took all the BC data, we would give
    # BC an unfair advantage. To be conservative, we should find the worst-case fraction of the other runs we take
    # (i.e. prefer 1/4 to 3/4), and take the same amount of BC data.

    for env_name in events_by_env_name_by_run_type_by_seed.keys():
        if 'BC' not in events_by_env_name_by_run_type_by_seed[env_name]:
            continue
        # Find non-BC run that we keep the least of
        min_frac = float('inf')
        events_with_min_frac = None
        for run_type in ['DRLHP', 'SDRLHP', 'SDRLHP-BC', 'RL']:
            if run_type not in events_by_env_name_by_run_type_by_seed[env_name]:
                continue
            for seed in events_by_env_name_by_run_type_by_seed[env_name][run_type].keys():
                events: dict = events_by_env_name_by_run_type_by_seed[env_name][run_type][seed][0]
                last_n_steps = events['policy_master/n_total_steps'][-1][1]
                frac = max_steps / last_n_steps
                if frac < min_frac:
                    min_frac = frac
                    events_with_min_frac = events

        # Find timestamp corresponding to max_steps
        steps = [tup[1] for tup in events_with_min_frac['policy_master/n_total_steps']]
        i = np.argmin(np.abs(np.array(steps) - max_steps))
        i = int(i)
        timestamp_for_max_steps = events_with_min_frac['policy_master/n_total_steps'][i][0]

        # Create fake steps such that we reach max_steps at that timestamp
        for events, _ in events_by_env_name_by_run_type_by_seed[env_name]['BC'].values():
            fake_steps = [(timestamp, timestamp / timestamp_for_max_steps * max_steps)
                          for timestamp, value in events['policy_master/n_updates']]
            events['policy_master/n_total_steps'] = fake_steps


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--runs_dir')
    group.add_argument('--test', action='store_true')
    parser.add_argument('--max_steps', type=float)
    parser.add_argument('--max_hours', type=float)
    parser.add_argument('--train_env_key', default='env_train')
    args = parser.parse_args()

    if args.test:
        sys.argv.pop(1)
        unittest.main()

    for f in glob.glob('*.png'):
        os.remove(f)

    events_by_env_name_by_run_type_by_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for run_dir in os.scandir(args.runs_dir):
        print(f"Reading events for {run_dir.name}...")
        events = read_all_events(run_dir.path)
        try:
            env_name, run_type, seed = parse_run_name(run_dir.name)
        except Exception as e:
            print(e)
            continue
        if run_type in ['DRLHP', 'SDRLHP', 'SDRLHP-BC']:
            filter_pretraining_events(run_dir.path, events)
        make_timestamps_relative_hours(events)
        events_by_env_name_by_run_type_by_seed[env_name][run_type][seed] = (events, run_dir.name)

    if args.max_steps:
        add_steps_to_bc_run(events_by_env_name_by_run_type_by_seed, args.max_steps)

    time_values_fn = partial(get_values_by_time, max_hours=args.max_hours)
    step_values_fn = partial(get_values_by_step, max_steps=args.max_steps)

    for env_name, events_by_run_type_by_seed in events_by_env_name_by_run_type_by_seed.items():
        print(f"Plotting {env_name}...")
        metrics = detect_metrics(env_name, args.train_env_key)
        for value_fn, x_type, x_label, x_lim in [
            (time_values_fn, 'time', 'Hours', args.max_hours),
            (step_values_fn, 'step', 'Total environment steps', args.max_steps)]:
            for metric_n, metric in enumerate(metrics):
                figure(metric_n)
                all_min_y = float('inf')
                all_max_y = -float('inf')
                for run_type_n, (run_type, events_by_seed) in enumerate(events_by_run_type_by_seed.items()):
                    try:
                        color = f"C{run_type_n}"
                        xs_list = []
                        ys_list = []
                        for events, run_dir in events_by_seed.values():
                            if metric.tag not in events:
                                print(f"Error: couldn't find metric '{metric.tag}' in run '{run_dir}'", file=sys.stderr)
                                exit(1)
                            xs, ys = value_fn(events, metric)
                            xs_list.append(xs)
                            ys_list.append(ys)
                        min_y, max_y = plot_averaged(xs_list, ys_list, metric.smoothing, metric.fillsmoothing, color,
                                                     run_type)
                        all_max_y = max_y if max_y > all_max_y else all_max_y
                        all_min_y = min_y if min_y < all_min_y else all_min_y
                    except KeyError:
                        # BC runs don't store steps
                        continue

                if all_min_y == float('inf'):
                    # If we didn't plot anything because we were plotting by steps and none of the runs logged steps
                    continue

                legend()
                xlabel(x_label)
                ylabel(metric.name)
                xlim(left=0)
                if x_lim is not None:
                    xlim(right=x_lim)
                ylim([all_min_y, all_max_y])

                escaped_env_name = env_name.replace(' ', '_').lower()
                escaped_metric_name = metric.name.replace(' ', '_').replace('.', '').lower()
                fig_filename = '{}_{}_by_{}.png'.format(escaped_env_name, escaped_metric_name, x_type)
                savefig(fig_filename, dpi=300, bbox_inches='tight')
            close('all')


if __name__ == '__main__':
    main()
