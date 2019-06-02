import itertools
import json
import multiprocessing
import os
import pickle
import random
import re
import shutil
import time
from threading import Thread
from typing import Dict

import easy_tf_log
from flask import request, render_template, send_from_directory, Blueprint

import global_variables
from rollouts import CompressedRollout
from utils import concatenate_and_write_videos_to
from web_app import web_globals
from web_app.utils import nocache, add_pref, get_n_rl_steps
from web_app.web_globals import _demonstration_rollouts, experience_dir, _policies, \
    _policy_rollouter, _demonstration_rollouts_dir, _reset_state_cache
from wrappers.util_wrappers import SaveEpisodeStats

"""
Bugs:
- Isn't resilient to multiple users giving demonstrations at the same time
  (e.g. it might happen that two users are given the same set of rollouts to compare)
"""

demonstrations_app = Blueprint('demonstrations', __name__)

n_rollouts_in_progress = multiprocessing.Value('i', 0)
n_trajectories_demonstrated = multiprocessing.Value('i', 0)
n_demonstrations_given = multiprocessing.Value('i', 0)
trajectory_for_group_dict = {}
logger = easy_tf_log.Logger()
logger.set_log_dir(_demonstration_rollouts_dir)
episode_stats_logger = None
n_rl_steps_at_last_pref = None


@demonstrations_app.route('/demonstrate', methods=['GET'])
def demonstrate():
    return render_template('demonstrate.html')


def get_metadatas():
    return [f for f in os.listdir(_demonstration_rollouts_dir) if re.match(r'^metadata.*\.json$', f)]


@demonstrations_app.route('/generate_rollouts', methods=['GET'])
def generate_rollouts():
    policies_str = request.args['policies']
    if not policies_str:
        policy_names_list = list(_policies.policies.keys())
    else:
        policy_names_list = policies_str.split(',')

    trajectory_serial = start_new_trajectory()
    group_serial = _policy_rollouter.generate_rollouts_from_reset(policy_names_list,
                                                                  softmax_temp=1.0)
    trajectory_for_group_dict[group_serial] = trajectory_serial

    return ""


@demonstrations_app.route('/get_rollouts', methods=['GET'])
def get_rollouts():
    global n_rl_steps_at_last_pref
    n_rl_steps = get_n_rl_steps()
    if n_rl_steps is not None and n_rl_steps_at_last_pref is not None:  # Maybe we haven't started training yet
        n_rl_steps_since_last_pref = n_rl_steps - n_rl_steps_at_last_pref
        if n_rl_steps_since_last_pref < global_variables.min_n_rl_steps_per_pref:
            return 'No rollouts available'

    rollout_groups = get_metadatas()
    if not rollout_groups:
        if n_rollouts_in_progress.value == 0:
            print("Error: No rollouts in progress?")  # TODO debugging; deleteme
            return json.dumps([])
        else:
            print("Waiting for rollouts...")
            while not rollout_groups:
                time.sleep(0.1)
                rollout_groups = get_metadatas()

    rollout_group = sorted(rollout_groups)[0]
    with open(os.path.join(_demonstration_rollouts_dir, rollout_group), 'r') as f:
        try:
            rollout_hashes = json.load(f)
        except Exception as e:
            print(os.listdir(_demonstration_rollouts_dir))  # TODO debugging; deletemd
            print(f"Exception while trying to read {rollout_group}")
            raise e
    rollouts = {}
    for rollout_hash in rollout_hashes:
        with open(os.path.join(_demonstration_rollouts_dir, rollout_hash + '.pkl'), 'rb') as f:
            rollout = pickle.load(f)
        rollouts[rollout_hash] = rollout

    rollout_dict = {rollout_hash_str: (rollout.generating_policy, rollout.vid_filename, rollout.rewards)
                    for rollout_hash_str, rollout in rollouts.items()}
    return json.dumps([rollout_group, rollout_dict])


def process_choice_and_generate_new_rollouts(rollouts: Dict[str, CompressedRollout], rollout_choice,
                                             group_name, trajectory_dir, trajectory_serial,
                                             policy_names, softmax_temp):
    global episode_stats_logger

    if rollout_choice != 'equal' and rollouts[rollout_choice].generating_policy == 'redo':
        continue_with_rollout = rollouts[rollout_choice]
        force_reset = False
    else:
        for h, r in rollouts.items():
            if r.generating_policy == 'redo':
                del rollouts[h]
                break

        if rollout_choice == 'none':
            continue_with_rollout_hash = random.sample(list(rollouts.keys()), 1)[0]
            continue_with_rollout = rollouts[continue_with_rollout_hash]
        elif rollout_choice == 'equal':
            for r1, r2 in itertools.combinations(rollouts.values(), 2):
                add_pref(r1, r2, (0.5, 0.5))
            continue_with_rollout_hash = random.sample(list(rollouts.keys()), 1)[0]
            continue_with_rollout = rollouts[continue_with_rollout_hash]
        else:
            chosen_rollout = rollouts[rollout_choice]
            add_demonstration_rollout(chosen_rollout)
            add_reset_state_from_end_of_rollout(chosen_rollout)
            rollouts_except_chosen = set(rollouts.values()) - {chosen_rollout}
            for other_rollout in rollouts_except_chosen:
                add_pref(chosen_rollout, other_rollout, (1.0, 0.0))
            continue_with_rollout_hash = rollout_choice
            continue_with_rollout = chosen_rollout
        print("Continuing with rollout {}".format(continue_with_rollout.hash))

        # write which rollout was chosen for this group in the trajectory file
        trajectory_filename = os.path.join(trajectory_dir, "trajectory_{}".format(trajectory_serial))
        timestamp = str(time.time())
        with open(trajectory_filename, 'a') as f:
            f.write(group_name + "," + continue_with_rollout_hash + "," + timestamp + "," + rollout_choice + "\n")

        trajectory_dir = get_trajectory_dir(trajectory_serial)
        trajectory_filename = os.path.join(trajectory_dir, "trajectory_{}".format(trajectory_serial))
        with open(trajectory_filename, 'r') as f:
            n_rollouts_in_this_demonstration = len(f.readlines())
        if web_globals._max_demonstration_length is not None and n_rollouts_in_this_demonstration == web_globals._max_demonstration_length:
            print("Reached maximum demonstration length ({}); resetting".format(web_globals._max_demonstration_length))
            force_reset = True
        else:
            force_reset = False

        if force_reset or continue_with_rollout.final_env_state.done:
            env = continue_with_rollout.final_env_state.env
            if episode_stats_logger is None:
                episode_stats_logger = SaveEpisodeStats(env,
                                                        os.path.join(_demonstration_rollouts_dir, 'demo_env'),
                                                        suffix='_demo')
            else:
                episode_stats_logger.set_env(env)
            episode_stats_logger.reset()  # trigger stats save

            trajectory_dir = get_trajectory_dir(trajectory_serial)
            vid_name = os.path.join(trajectory_dir, 'demonstrated_trajectory_{}.mp4'.format(trajectory_serial))
            chosen_rollout_vid_paths = get_chosen_rollout_videos_for_trajectory(trajectory_serial)
            concatenate_and_write_videos_to(chosen_rollout_vid_paths, vid_name)
            trajectory_serial = start_new_trajectory()
            n_trajectories_demonstrated.value += 1
            logger.logkv('demonstrations/n_trajectories_demonstrated', n_trajectories_demonstrated.value)

    group_serial = _policy_rollouter.generate_rollout_group(continue_with_rollout.final_env_state,
                                                            continue_with_rollout.generating_policy,
                                                            policy_names,
                                                            softmax_temp,
                                                            force_reset)
    trajectory_for_group_dict[group_serial] = trajectory_serial

    n_rollouts_in_progress.value -= 1

@demonstrations_app.route('/choose_rollout', methods=['GET'])
def choose_rollout():
    global n_rl_steps_at_last_pref

    group_name = request.args['group']
    chosen_rollout_hash_str = request.args['hash']
    policies_str = request.args['policies']

    try:
        softmax_temp = request.args['softmax_temp']
    except:
        softmax_temp = 1

    if policies_str:
        policy_names = policies_str.split(',')
    else:
        policy_names = list(_policies.policies.keys())

    if group_name not in os.listdir(_demonstration_rollouts_dir):
        return f"Error: group '{group_name}' doesn't exist"

    group_filename = os.path.join(_demonstration_rollouts_dir, group_name)
    try:
        with open(group_filename, 'r') as f:
            rollout_group_hash_strs = json.load(f)
    except Exception as e:
        print(f"Exception while trying to read '{group_filename}':")  # TODO debugging; deleteme
        return e
    if not chosen_rollout_hash_str in rollout_group_hash_strs and chosen_rollout_hash_str != 'none' and chosen_rollout_hash_str != 'equal':
        msg = f"Error: rollout '{chosen_rollout_hash_str}' not in group '{group_name}'"
        print(msg)
        return msg

    group_serial = re.match(r'metadata_(.*)\.json', group_name)[1]
    trajectory_serial = trajectory_for_group_dict[group_serial]
    trajectory_dir = get_trajectory_dir(trajectory_serial)
    shutil.move(group_filename, trajectory_dir)

    rollouts = {}  # type: Dict[str, CompressedRollout]
    for hash_str in rollout_group_hash_strs:
        file_prefix = os.path.join(_demonstration_rollouts_dir, hash_str)
        pkl_name = file_prefix + '.pkl'
        vid_name = file_prefix + '.mp4'

        with open(pkl_name, 'rb') as f:
            rollouts[hash_str] = pickle.load(f)

        shutil.move(pkl_name, trajectory_dir)
        shutil.move(vid_name, trajectory_dir)

    n_rollouts_in_progress.value += 1
    print("No. rollouts in progress:", n_rollouts_in_progress)  # TODO debugging; deleteme
    Thread(target=lambda: process_choice_and_generate_new_rollouts(rollouts, chosen_rollout_hash_str,
                                                                   group_name, trajectory_dir, trajectory_serial,
                                                                   policy_names, softmax_temp)).start()

    n_rl_steps_at_last_pref = get_n_rl_steps()
    if n_rl_steps_at_last_pref is not None:
        logger.logkv('demonstrations/n_rl_steps_at_last_pref', n_rl_steps_at_last_pref)

    return ""

def add_demonstration_rollout(rollout):
    _demonstration_rollouts[rollout.hash] = rollout
    _demonstration_rollouts.save(os.path.join(experience_dir, 'demonstration_rollouts.pkl'))
    print("Added rollout {} as a demonstration rollout".format(rollout.hash))
    n_demonstrations_given.value += 1
    logger.logkv('demonstrations/n_demonstrations', len(_demonstration_rollouts))
    logger.logkv('demonstrations/added_demonstrations', n_demonstrations_given.value)


def add_reset_state_from_end_of_rollout(rollout: CompressedRollout):
    print("Adding a reset state from the end of rollout {}".format(rollout.hash))
    reset_state = rollout.final_env_state
    _reset_state_cache.queue_from_demonstrations.put(reset_state)


def get_chosen_rollout_videos_for_trajectory(trajectory_serial):
    trajectory_dir = get_trajectory_dir(trajectory_serial)
    trajectory_filename = os.path.join(trajectory_dir,
                                       "trajectory_{}".format(trajectory_serial))
    with open(trajectory_filename, 'r') as f:
        lines = [x.strip() for x in f.readlines()]

    chosen_rollout_vid_paths = []
    for line in lines:
        group_serial, continue_rollout_hash, timestamp, _ = line.split(',')
        chosen_rollout_vid_path = os.path.join(trajectory_dir, continue_rollout_hash + '.mp4')
        chosen_rollout_vid_paths.append(chosen_rollout_vid_path)

    return chosen_rollout_vid_paths

def start_new_trajectory():
    serial = str(time.time())
    trajectory_dir = get_trajectory_dir(serial)
    os.makedirs(trajectory_dir)
    return serial

def get_trajectory_dir(trajectory_serial):
    return os.path.join(_demonstration_rollouts_dir, "trajectory_{}".format(trajectory_serial))

@demonstrations_app.route('/get_rollout_video', methods=['GET'])
@nocache
def get_rollout_video():
    filename = request.args['filename']
    return send_from_directory(_demonstration_rollouts_dir, filename)


@demonstrations_app.route("/reset_demonstration_env", methods=['GET'])
def reset_env():
    _policy_rollouter.reset()
    return ""
