import os
import time
from collections import defaultdict
from enum import Enum

import easy_tf_log
import numpy as np
import tensorflow as tf

import global_variables

logger = None
log_dir = None
n_rl_steps_at_last_event = defaultdict(lambda: 0)


class Stats:
    def __init__(self, init_time, n_rl_steps_since_last_event_list):
        self.init_time = init_time
        self.n_rl_steps_since_last_list = n_rl_steps_since_last_event_list


event_stats = defaultdict(lambda: Stats(init_time=0, n_rl_steps_since_last_event_list=[]))


class EventType(Enum):
    INTERACTION = 0
    REWARD_PREDICTOR_15_BATCHES = 1


# ID necessary because we might have to initialize different throttlers for difference processes
def init(_log_dir, log_dir_id):
    global logger, log_dir
    if logger is not None:
        raise Exception("Throttled already initialized")
    log_dir = _log_dir
    logger = easy_tf_log.Logger(os.path.join(_log_dir, f'throttler_{log_dir_id}'))


def get_n_rl_steps_per_event(event_type):
    if event_type == EventType.INTERACTION:
        return global_variables.n_rl_steps_per_interaction
    elif event_type == EventType.REWARD_PREDICTOR_15_BATCHES:
        # lowest_seen_n_batches_per_second = 4
        # expected_n_rl_steps_per_second = 800
        # steps_per_batch = expected_n_rl_steps_per_second / lowest_seen_n_batches_per_second
        # steps_per_15_batches = steps_per_batch * 15
        steps_per_15_batches = 300
        return steps_per_15_batches


def mark_event(event_type):
    print("Mark at", time.time())
    n_rl_steps = get_n_rl_steps()
    if n_rl_steps is None:
        return

    n_rl_steps_since_last_event = n_rl_steps - n_rl_steps_at_last_event[event_type]
    # This gets called for every reward predictor training step, so we need to be careful not to log too often
    stats = event_stats[event_type]
    stats.n_rl_steps_since_last_list.append(n_rl_steps_since_last_event)
    seconds_since_last_log = time.time() - stats.init_time
    if seconds_since_last_log > 60:
        frac = np.array(stats.n_rl_steps_since_last_list) / get_n_rl_steps_per_event(event_type)
        for aggregation_str, aggregration_fn in [('max', max), ('min', min)]:
            logger.logkv(f'throttler/n_rl_steps_since_last_{aggregation_str}_{str(event_type).lower()}',
                         aggregration_fn(stats.n_rl_steps_since_last_list))
            logger.logkv(f'throttler/target_frac_{aggregation_str}_{str(event_type).lower()}',
                         aggregration_fn(frac))
        event_stats[event_type] = Stats(init_time=time.time(), n_rl_steps_since_last_event_list=[])

    n_rl_steps_at_last_event[event_type] = n_rl_steps


def read_events_file(events_filename):
    events = {}
    try:
        for event in tf.train.summary_iterator(events_filename):
            for value in event.summary.value:
                if value.tag not in events:
                    events[value.tag] = []
                events[value.tag].append((event.wall_time, event.step, value.simple_value))
    except Exception as e:
        print(f"While reading '{events_filename}':", e)
    return events


def get_n_rl_steps():
    global log_dir
    fname = os.path.join(log_dir, 'policy_master', 'n_total_steps.txt')
    if not os.path.exists(fname):
        # If we haven't started training yet (i.e. we're still pretraining)
        return None
    with open(fname, 'r') as f:
        n = int(f.read())
    return n


def check_throttle(event_type: EventType):
    n_rl_steps_per_event = get_n_rl_steps_per_event(event_type)
    # 0 => don't throttle
    if n_rl_steps_per_event == 0:
        return

    n_rl_steps = get_n_rl_steps()
    if n_rl_steps is None:
        return

    n_rl_steps_since_last_event = n_rl_steps - n_rl_steps_at_last_event[event_type]
    if n_rl_steps_since_last_event < n_rl_steps_per_event:
        return True
    else:
        return False


def throttle_sleep(event_type: EventType):
    while check_throttle(event_type):
        time.sleep(1e-3)
