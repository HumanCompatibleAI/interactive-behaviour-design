import os
import time
from collections import defaultdict
from enum import Enum

import easy_tf_log

import global_variables


class EventType(Enum):
    INTERACTION = 0
    REWARD_PREDICTOR_10_STEPS = 1


logger = None
log_dir = None
n_rl_steps_at_last_event = defaultdict(lambda: 0)


# log_dir_is necessary because we might have to initialize different throttlers for difference processes
def init(_log_dir, log_dir_id):
    global logger, log_dir
    if logger is not None:
        raise Exception("Throttled already initialized")
    log_dir = _log_dir
    logger = easy_tf_log.Logger(os.path.join(_log_dir, f'throttler_{log_dir_id}'))


def get_n_rl_steps_per_event(event_type):
    if event_type == EventType.INTERACTION:
        return global_variables.n_rl_steps_per_interaction
    elif event_type == EventType.REWARD_PREDICTOR_10_STEPS:
        desired_n_reward_predictor_steps_per_second = 50
        expected_n_rl_steps_per_second = 800
        n_rl_steps_per_reward_predictor_step = expected_n_rl_steps_per_second / desired_n_reward_predictor_steps_per_second
        n_rl_steps_per_10_reward_predictor_steps = n_rl_steps_per_reward_predictor_step * 10
        return n_rl_steps_per_10_reward_predictor_steps


def mark_event(event_type):
    n_rl_steps = get_n_rl_steps()
    if n_rl_steps is None:
        return
    log_stats(event_type, n_rl_steps)
    n_rl_steps_at_last_event[event_type] = n_rl_steps


def log_stats(event_type, n_rl_steps):
    n_rl_steps_since_last_event = n_rl_steps - n_rl_steps_at_last_event[event_type]
    n_target = get_n_rl_steps_per_event(event_type)
    # 0 => don't throttle
    if n_target == 0:
        return
    frac = n_rl_steps_since_last_event / n_target
    logger.logkv(f'throttler/{event_type.name}/target_frac', frac)


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
        return False

    n_rl_steps = get_n_rl_steps()
    if n_rl_steps is None:
        return False

    n_rl_steps_since_last_event = n_rl_steps - n_rl_steps_at_last_event[event_type]
    if n_rl_steps_since_last_event < n_rl_steps_per_event:
        return True
    else:
        return False


def throttle_sleep(event_type: EventType):
    while check_throttle(event_type):
        time.sleep(1e-3)
