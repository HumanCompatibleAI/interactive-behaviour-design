#!/usr/bin/env python3

"""
Plot metrics by time and by step, plotting a solid line showing the mean of all seeds and a shaded region
showing one standard error between seeds.

# Run me with --runs_dir pointing to a directory containing runs like:
#   fetch-0-drlhp_foobar
# where 'fetch' is the environment name, '0' is the seed, 'drlhp' is the run type, and foobar is ignored
"""

import argparse
import fnmatch
import glob
import locale
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
from matplotlib import colors
from matplotlib.pyplot import close, fill_between, title
from tensorflow.python.util import deprecation

matplotlib.use('Agg')

from pylab import plot, xlabel, ylabel, figure, legend, savefig, grid, ylim, xlim, ticklabel_format, show

deprecation._PRINT_DEPRECATION_WARNINGS = False

# Get thousands separated by commas
locale.format_string = partial(locale.format_string, grouping=True)


# locale.setlocale(locale.LC_ALL, 'en_GB.utf8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs_dir', nargs='?')
    parser.add_argument('--max_steps', type=float)
    parser.add_argument('--max_hours', type=float)
    parser.add_argument('--train_env_key', default='env_train')
    parser.add_argument('--test', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--with_individual_seeds', action='store_true')
    group.add_argument('--individual_seeds_only', action='store_true')
    parser.add_argument('--smooth_individual_seeds', action='store_true')
    args = parser.parse_args()

    if args.test:
        sys.argv.pop(1)
        unittest.main()

    for f in glob.glob('*.png'):
        os.remove(f)

    events_by_env_name_by_run_type_by_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for run_dir in [path for path in os.scandir(args.runs_dir) if os.path.isdir(path.path)]:
        print(f"Reading events for {run_dir.name}...")
        events = read_all_events(run_dir.path)
        try:
            env_name, run_type, seed = parse_run_name(run_dir.name)
        except Exception as e:
            print(e)
            continue
        if 'DRLHP' in run_type:
            filter_pretraining_events(run_dir.path, events)
        drop_first_event(events)
        make_timestamps_relative_hours(events)
        events_by_env_name_by_run_type_by_seed[env_name][run_type][seed] = (events, run_dir.name)

    if args.max_steps:
        add_steps_to_bc_run(events_by_env_name_by_run_type_by_seed, args.max_steps)

    for env_name, events_by_run_type_by_seed in events_by_env_name_by_run_type_by_seed.items():
        plot_env(args, env_name, events_by_run_type_by_seed)

def downsample_to_changes(timestamps, values):
    downsampled = []
    last_v = None
    for t, v in zip(timestamps, values):
        if last_v is None or v != last_v:
            downsampled.append((t, v))
        last_v = v
    timestamps, values = list(zip(*downsampled))
    return timestamps, values

def plot_env(args, env_name, events_by_run_type_by_seed):
    print(f"Plotting {env_name}...")
    metrics = detect_metrics(env_name, args.train_env_key)
    time_values_fn = partial(get_values_by_time, max_hours=args.max_hours)
    steps_value_fn = partial(get_values_by_step, max_steps=args.max_steps)
    plot_modes = [
        (time_values_fn, 'time', 'Hours', args.max_hours),
        (steps_value_fn, 'steps', 'Total environment steps', args.max_steps),
        (get_values_by_n_human_interactions, 'interactions', 'No. human interactions', None)
    ]
    for value_fn, x_type, x_label, x_lim in plot_modes:
        for metric_n, metric in enumerate(metrics):
            print(f"Metric '{metric.tag}'")
            figure(metric_n)
            all_min_y = float('inf')
            all_max_y = -float('inf')
            for run_type_n, (run_type, events_by_seed) in enumerate(events_by_run_type_by_seed.items()):
                if run_type in ['BC', 'BCNP'] and value_fn == steps_value_fn:
                    continue
                if run_type == 'RL' and value_fn == get_values_by_n_human_interactions:
                    continue
                print(f"Run type '{run_type}'")

                xs_list, ys_list = get_values(events_by_seed, metric, run_type, value_fn)
                color = colors.to_rgba(f"C{run_type_n}")
                n_seeds = len(xs_list)
                if args.with_individual_seeds or args.individual_seeds_only:
                    if args.individual_seeds_only:
                        linewidth = 1
                        labels = [f'{run_type}-{seed}' for seed in range(n_seeds)]
                    else:
                        linewidth = 0.5
                        labels = [None] * n_seeds  # Labels will get set by plot_averaged
                    individual_run_opacities = np.linspace(0.3, 1.0, n_seeds)
                    for seed in range(n_seeds):
                        timestamps, values = xs_list[seed], ys_list[seed]
                        if args.smooth_individual_seeds:
                            values = smooth_values_exponential(values, 0.97)
                        plot(timestamps, values, color=color, linewidth=linewidth,
                             alpha=individual_run_opacities[seed], label=labels[seed])
                if not args.individual_seeds_only:
                    plot_averaged(xs_list, ys_list,
                                  metric.window_size, metric.fill_window_size,
                                  color, run_type)

                all_min_y = min(all_min_y, np.min([np.min(ys) for ys in ys_list]))
                all_max_y = max(all_max_y, np.max([np.max(ys) for ys in ys_list]))

            grid(True)
            ticklabel_format(axis='x', style='scientific', scilimits=(0, 5), useLocale=True)
            xlabel(x_label)
            ylabel(metric.name)
            xlim(left=0)
            if x_lim is not None:
                xlim(right=x_lim)
            y_range = all_max_y - all_min_y
            all_min_y -= y_range / 10
            all_max_y += y_range / 10
            ylim([all_min_y, all_max_y])
            title(env_name)
            legend(bbox_to_anchor=(1.0, 0.5), loc='center left')

            escaped_env_name = escape_name(env_name)
            escaped_metric_name = escape_name(metric.name)
            fig_filename = '{}_{}_by_{}.png'.format(escaped_env_name, escaped_metric_name, x_type)
            savefig(fig_filename, dpi=300, bbox_inches='tight')

        close('all')


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
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    events_in_each_file = pool.map(read_events_file, events_files)
    all_events = {}
    for events in events_in_each_file:
        all_events.update(events)
    return all_events


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


def smooth_values(values, window_size):
    window = np.ones(window_size)
    actual_window_sizes = np.convolve(np.ones(len(values)), window, 'same')
    smoothed_values = np.convolve(values, window, 'same') / actual_window_sizes
    return smoothed_values


def smooth_values_exponential(values, smoothing):
    smoothed_values = [values[0]]
    for v in values[1:]:
        smoothed_values.append(smoothing * smoothed_values[-1] + (1 - smoothing) * v)
    return smoothed_values


def interpolate_to_common_xs(timestamp_y_tuples, timestamp_x_tuples):
    x_timestamps, xs = zip(*timestamp_x_tuples)
    y_timestamps, ys = zip(*timestamp_y_tuples)

    if len(timestamp_x_tuples) < len(timestamp_y_tuples):
        # Use x timestamps for interpolation
        xs = xs
        ys = interpolate_values(timestamp_y_tuples, x_timestamps)
    elif len(timestamp_y_tuples) < len(timestamp_x_tuples):
        # Use y timestamps for interpolation
        xs = interpolate_values(timestamp_x_tuples, y_timestamps)
        ys = ys
    else:
        pass

    # interpolate_values uses NaN to signal "couldn't interpolate this value"
    # (because we didn't have data at the beginning or end); let's remove those points
    drop_idxs = []
    for i in range(len(xs)):
        if np.isnan(xs[i]):
            drop_idxs.append(i)
    xs = [xs[i] for i in range(len(xs)) if i not in drop_idxs]
    ys = [ys[i] for i in range(len(ys)) if i not in drop_idxs]

    return xs, ys


def find_training_start(run_dir):
    # The auto train script exits once training has started properly
    train_log_path = os.path.join(run_dir, 'auto_train.log')
    if os.path.exists(train_log_path):
        return os.path.getmtime(train_log_path)
    else:
        raise Exception()


M = namedtuple('M', 'tag name window_size fill_window_size')


def detect_metrics(env_name, train_env_key):
    metrics = []
    if env_name == 'Lunar Lander':
        metrics.append(M(f'{train_env_key}/reward_sum', 'Episode reward', 100, 100))
        metrics.append(M(f'{train_env_key}/crashes', 'Crash rate', 100, 100))
        metrics.append(M(f'{train_env_key}/successful_landing_rate', 'Successful landing rate', 100, 100))
    if env_name == 'Seaquest':
        metrics.append(M(f'{train_env_key}/reward_sum', 'Episode reward', 100, 100))
        metrics.append(M(f'{train_env_key}/n_diver_pickups', 'Diver pickups per episode', 500, 500))
    if env_name == 'Breakout':
        metrics.append(M(f'{train_env_key}/reward_sum', 'Episode reward', 100, 100))
    if env_name == 'Enduro':
        metrics.append(M(f'{train_env_key}/reward_sum', 'Episode reward', 100, 100))
    if env_name == 'Fetch pick-and-place':
        metrics.append(M(f'env_test/ep_frac_aligned_with_block', 'Fraction of episode aligned with block', 100, 100))
        metrics.append(M(f'env_test/ep_frac_gripping_block', 'Fraction of episode gripping block', 100, 100))
        metrics.append(M(f'env_test/success_rate', 'Success rate (end of episode)', 10, 10))
        metrics.append(M(f'env_test/success_near_end_rate', 'Success rate (near end of episode)', 10, 10))
        metrics.append(M(f'env_test/success_partial_rate', 'Success rate (anywhere in episode)', 10, 10))
    if env_name == 'Fetch reach':
        metrics.append(M(f'env_test/success_rate', 'Success rate (end of episode)', 10, 10))
        metrics.append(M(f'env_test/success_near_end_rate', 'Success rate (near end of episode)', 10, 10))
        metrics.append(M(f'env_test/success_partial_rate', 'Success rate (anywhere in episode)', 10, 10))
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


def plot_averaged(xs_list, ys_list, window_size, fill_window_size, color, label):
    # for ys in ys_list:
    #     if len(ys) < window_size:
    #         print(f"Error: window_size={window_size} but we only have {len(ys)} values", file=sys.stderr)
    #         exit(1)

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
    # plot_width_pixels = 1500  # determined by manually checking the figure
    # xs_downsampled = None
    # ys_downsampled_list = []
    # for ys in ys_list:
    #     xs_downsampled, ys_downsampled = downsample(all_xs, ys, plot_width_pixels)
    #     xs_downsampled, ys_downsampled = all_xs, ys
    #     assert len(ys_downsampled) == len(xs_downsampled)
    #     ys_downsampled_list.append(ys_downsampled)

    mean_ys = np.mean(ys_list, axis=0)  # Average across seeds
    if len(ys_list[0]) > window_size:
        smoothed_mean_ys = smooth_values(mean_ys, window_size)
    else:
        smoothed_mean_ys = mean_ys
    plot(all_xs, smoothed_mean_ys, color=color, label=label, alpha=0.9)

    if fill_window_size is not None:
        std = np.std(ys_list, axis=0)
        smoothed_std = np.array(smooth_values(std, fill_window_size))
        lower = smoothed_mean_ys - smoothed_std
        upper = smoothed_mean_ys + smoothed_std
        fill_between(all_xs, lower, upper, color=color, alpha=0.2)
        min_val, max_val = np.min(lower), np.max(upper)
    else:
        min_val, max_val = np.min(smoothed_mean_ys), np.max(smoothed_mean_ys)

    return min_val, max_val


def parse_run_name(run_dir):
    match = re.search(r'([^-]*)-([\d]*)-([^_]*)_', run_dir)  # e.g. fetch-0-drlhp_foobar
    if match is None:
        raise Exception(f"Couldn't parse run name '{run_dir}'")
    env_shortname = match.group(1)
    seed = match.group(2)
    run_type = match.group(3)

    env_shortname_to_env_name = {
        'fetchpp': 'Fetch pick-and-place',
        'fetchr': 'Fetch reach',
        'lunarlander': 'Lunar Lander',
        'enduro': 'Enduro',
        'breakout': 'Breakout',
        'seaquest': 'Seaquest'
    }
    if env_shortname not in env_shortname_to_env_name:
        raise Exception(f"Error: unsure of full env name for '{env_shortname}'")
    env_name = env_shortname_to_env_name[env_shortname]
    run_type = run_type.replace('-fetchreach', '')
    run_type = run_type.upper()

    return env_name, run_type, seed


def filter_pretraining_events(run_dir, events):
    try:
        training_start_timestamp = find_training_start(run_dir)
    except:
        # For runs where we didn't pretrain
        return
    tags = list(events.keys())
    for tag in tags:
        events[tag] = [(t, v) for t, v in events[tag] if t >= training_start_timestamp]
        if not events[tag]:
            del events[tag]
    # Reset the steps to start from 0 after the pretraining period
    first_step = events['policy_master/n_total_steps'][0][1]
    events['policy_master/n_total_steps'] = [(t, step - first_step)
                                             for t, step in events['policy_master/n_total_steps']]


def get_values_by_step(events, metric, run_type, max_steps):
    steps, values = interpolate_to_common_xs(events[metric.tag], events['policy_master/n_total_steps'])
    if max_steps:
        values = np.extract(np.array(steps) < max_steps, values)
        steps = np.extract(np.array(steps) < max_steps, steps)
    return steps, values


def combine_counters(timestamp_count_tuples_1, timestamp_count_tuples_2):
    check_increasing(timestamp_count_tuples_1)
    check_increasing(timestamp_count_tuples_2)

    timestamps_1 = [tup[0] for tup in timestamp_count_tuples_1]
    counts_1 = [tup[1] for tup in timestamp_count_tuples_1]
    deltas_1 = np.array(counts_1) - np.array([0] + counts_1[:-1])
    timestamp_delta_tuples_1 = list(zip(timestamps_1, deltas_1))

    timestamps_2 = [tup[0] for tup in timestamp_count_tuples_2]
    counts_2 = [tup[1] for tup in timestamp_count_tuples_2]
    deltas_2 = np.array(counts_2) - np.array([0] + counts_2[:-1])
    timestamp_delta_tuples_2 = list(zip(timestamps_2, deltas_2))

    all_tuples = timestamp_delta_tuples_1 + timestamp_delta_tuples_2
    all_tuples.sort(key=lambda tup: tup[0])
    combined_tuples = []
    combined_count = 0
    for tup in all_tuples:
        timestamp, count = tup
        combined_count += tup[1]
        combined_tuples.append((timestamp, combined_count))

    return combined_tuples


def check_increasing(timestamp_value_tuples):
    for i in range(1, len(timestamp_value_tuples)):
        tup_after = timestamp_value_tuples[i]
        tup_before = timestamp_value_tuples[i - 1]
        assert tup_after[1] >= tup_before[1], (tup_before, tup_after)


class TestCombineCounterEvents(unittest.TestCase):
    def test(self):
        counts1 = [(0, 10),
                   (1, 20),
                   (2, 30)]
        counts2 = [(0.5, 5),
                   (1.5, 6),
                   (2.5, 7)]
        combined = combine_counters(counts1, counts2)
        self.assertEqual(combined, [(0, 10),
                                    (0.5, 15),
                                    (1, 25),
                                    (1.5, 26),
                                    (2, 36),
                                    (2.5, 37)])


def get_values_by_n_human_interactions(events, metric, run_type):
    if run_type == 'DRLHP':
        n_interactions = events['pref_db/added_prefs']
    elif run_type in ['SDRLHP', 'SDRLHPNP', 'BC', 'BCNP']:
        n_interactions = events['demonstrations/added_demonstrations']
    elif run_type in ['SDRLHP-DRLHP', 'SDRLHPNP-DRLHP']:
        n_interactions = combine_counters(events['pref_db/added_prefs'], events['demonstrations/added_demonstrations'])
    else:
        raise Exception(f"Unsure which tag represents no. human interactions for run type '{run_type}'")
    xs, ys = interpolate_to_common_xs(events[metric.tag], n_interactions)
    return xs, ys


def get_values_by_time(events, metric, run_type, max_hours):
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
        for run_type in ['DRLHP', 'DRLHPD', 'SDRLHP', 'SDRLHPNP', 'SDRLHP-BC', 'RL']:
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


def drop_first_event(events):
    """
    Not sure why, but the first value is sometimes 0 or inf. Let's drop it.
    """
    for key in events:
        events[key] = events[key][1:]


def get_values(events_by_seed, metric, run_type, value_fn):
    xs_list = []
    ys_list = []
    for seed, (events, run_dir) in events_by_seed.items():
        if metric.tag not in events:
            print(f"Error: couldn't find metric '{metric.tag}' in run '{run_dir}'", file=sys.stderr)
            exit(1)
        xs, ys = value_fn(events, metric, run_type)
        xs_list.append(xs)
        ys_list.append(ys)
    return xs_list, ys_list


def escape_name(name):
    return name.replace(' ', '_').replace('.', '').replace('(', '').replace(')', '').lower()


if __name__ == '__main__':
    main()
