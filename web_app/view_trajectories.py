import json
import os
import pickle

from flask import request, render_template, send_from_directory, Blueprint

from web_app.utils import nocache
from web_app.web_globals import _demonstration_rollouts_dir
from web_app.demonstrations import get_trajectory_dir

view_trajectories_app = Blueprint('view_trajectories', __name__)

rollout_group_to_trajectory = {} #rollout group metadata filename : trajectory that it belongs to
@view_trajectories_app.route('/view_trajectories', methods=['GET'])
def demonstrate():
    return render_template('view_trajectories.html')

def get_trajectories():
    trajectory_dirs = [f for f in os.listdir(_demonstration_rollouts_dir) if 'trajectory' in f]
    trajectories = []
    for tr_dir_name in trajectory_dirs:
        traj_path = os.path.join(_demonstration_rollouts_dir, tr_dir_name)
        if ('demonstrated_' + tr_dir_name + '.mp4') in os.listdir(traj_path):
            #make sure to only include full trajectories (i.e. those with a 'demonstrated_trajectory_{}.mp4' file)
            trajectories.append(tr_dir_name)
    return trajectories

def get_trajectory_hash(f):
    """
    :param f: filename of trajectory (e.g. "trajectory_{hash}")
    :return: the hash of the trajectory (e.g. {hash} for "trajectory_{hash}")
    """
    prefix = "trajectory_"
    return int(f[len(prefix):])

@view_trajectories_app.route('/get_trajectory_list', methods=['GET'])
def get_trajectory_list():
    return json.dumps(get_trajectories())

@view_trajectories_app.route('/get_rollout_groups_for_trajectory', methods=['GET'])
def get_rollout_groups_for_trajectory():
    trajectory_name = request.args['trajectory_name']
    trajectory_dir_path = os.path.join(_demonstration_rollouts_dir, trajectory_name)
    trajectory_file_path = os.path.join(trajectory_dir_path, trajectory_name)
    with open(trajectory_file_path, 'r') as f:
        rollout_groups_with_chosen_rollout = f.readlines()
    rollout_groups_with_chosen_rollout = [x.strip() for x in
                                          rollout_groups_with_chosen_rollout]  # strip whitespace at end

    rollout_dict = {} #metadata filename : chosen rollout hash
    for rgwchr in rollout_groups_with_chosen_rollout:
        metadata_filename = rgwchr.split(',')[0]
        chosen_rollout_hash = rgwchr.split(',')[3]
        rollout_dict[metadata_filename] = chosen_rollout_hash
        rollout_group_to_trajectory[metadata_filename] = trajectory_name

    return json.dumps(rollout_dict)

@view_trajectories_app.route('/get_rollouts_for_group', methods=['GET'])
def get_rollouts_for_group():
    metadata_filename = request.args['metadata_filename']
    trajectory_dir = rollout_group_to_trajectory[metadata_filename]
    trajectory_path = os.path.join(_demonstration_rollouts_dir, trajectory_dir)
    with open(os.path.join(trajectory_path, metadata_filename), 'r') as f:
        try:
            rollout_hashes = json.load(f)
        except Exception as e:
            print(f"Exception while trying to read {metadata_filename}")
            raise e
    rollouts = {}
    for rollout_hash in rollout_hashes:
        with open(os.path.join(trajectory_path, rollout_hash + '.pkl'), 'rb') as f:
            rollout = pickle.load(f)
        rollouts[rollout_hash] = rollout

    rollout_dict = {rollout_hash_str: (rollout.generating_policy, rollout.vid_filename, rollout.rewards)
                    for rollout_hash_str, rollout in rollouts.items()}
    return json.dumps(rollout_dict)

@view_trajectories_app.route('/get_video', methods=['GET'])
@nocache
def get_rollout_video():
    trajectory_dir = request.args['trajectory']
    filename = request.args['filename']
    trajectory_path = os.path.join(_demonstration_rollouts_dir, trajectory_dir)
    return send_from_directory(trajectory_path, filename)

@view_trajectories_app.route('/get_policy_names', methods=['GET'])
def get_policy_names():
    trajectory_prefix = "trajectory_"
    trajectory_dirs = [dir for dir in os.listdir(_demonstration_rollouts_dir) if trajectory_prefix in dir]
    trajectory_dirs.sort()
    trajectory_path = os.path.join(_demonstration_rollouts_dir, trajectory_dirs[-2]) #2nd last since last may not contain any info

    metadata_prefix = "metadata_"
    metadata_filenames = [filename for filename in os.listdir(trajectory_path) if metadata_prefix in filename]
    metadata_filenames.sort()
    last_metadata_filename = metadata_filenames[-1]

    with open(os.path.join(trajectory_path, last_metadata_filename), 'r') as f:
        try:
            rollout_hashes = json.load(f)
        except Exception as e:
            print(f"Exception while trying to read {last_metadata_filename}")
            raise e
    rollouts = {}
    for rollout_hash in rollout_hashes:
        with open(os.path.join(trajectory_path, rollout_hash + '.pkl'), 'rb') as f:
            rollout = pickle.load(f)
        rollouts[rollout_hash] = rollout

    policies = set()
    for rollout_hash_str, rollout in rollouts.items():
        policies.add(rollout.generating_policy)
    return list(policies)

@view_trajectories_app.route('/get_timestamp_deltas_and_policy_frequencies', methods=['GET'])
def get_timestamp_deltas_and_policy_frequencies():
    trajectory_prefix = "trajectory_"
    trajectory_dirs = [dir for dir in os.listdir(_demonstration_rollouts_dir) if trajectory_prefix in dir]

    chosen_policies = []
    timestamp_deltas = []
    for trajectory_name in trajectory_dirs:  # loop through each trajectory
        trajectory_dirname = os.path.join(_demonstration_rollouts_dir, trajectory_name)
        trajectory_filename = os.path.join(trajectory_dirname, trajectory_name)

        if os.path.exists(trajectory_filename):
            with open(trajectory_filename, 'r') as f:
                lines = [x.strip() for x in f.readlines()]

            timestamps = []
            for line in lines:
                group_serial, chosen_rollout_hash, timestamp = line.split(',')
                with open(os.path.join(trajectory_dirname, chosen_rollout_hash + '.pkl'), 'rb') as f:
                    rollout = pickle.load(f)
                chosen_policy = rollout.generating_policy
                chosen_policies += [chosen_policy]
                timestamps += [timestamp]

            # calculate deltas for this one trajectory
            # deltas between trajectories could be meaningless if user took many breaks, like me :D
            for i in range(len(timestamps) - 1):
                t1 = timestamps[i + 1]
                t2 = timestamps[i]
                delta = float(t1) - float(t2)
                timestamp_deltas += [delta]
        else:
            print("Path: {} does not exist in get_timestamp_deltas_and_policy_frequencies()".format(trajectory_filename))

    # clean outliers in timestamp_deltas
    threshold = 20  # seconds
    timestamp_deltas = [td for td in timestamp_deltas if td < threshold]

    # calculate chosen policy frequencies after every n rollouts
    every_n = 10
    policy_names = get_policy_names()
    print(policy_names)
    policy_frequencies = {policy_name: [0]
                          for policy_name in policy_names}
    for i in range(len(chosen_policies)):
        if i > 0 and i % every_n == 0: #start new frequency count
            for policy_name in policy_frequencies:
                policy_frequencies[policy_name][-1] /= every_n #divide previous count to get frequency
                policy_frequencies[policy_name] += [0] #new count
        chosen_policy = chosen_policies[i]
        policy_frequencies[chosen_policy][-1] += 1
    if i % every_n != 0:
        for policy_name in policy_frequencies:
            policy_frequencies[policy_name][-1] /= i % every_n  # divide last count to get frequency

    return json.dumps([timestamp_deltas, policy_frequencies])