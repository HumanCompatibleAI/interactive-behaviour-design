import glob
import json
import multiprocessing
import os
import pickle
import random
import shutil
import time
from itertools import combinations

import easy_tf_log
from flask import Blueprint, render_template, request, send_from_directory
from gym.utils import atomic_write

import global_variables
import throttler
from rollouts import CompressedRollout
from throttler import check_throttle
from utils import make_small_change, save_video
from web_app import web_globals
from web_app.utils import add_pref, nocache

comparisons_app = Blueprint('comparisons', __name__)
logger = None
n_rl_steps_at_last_pref = None
labelled_seg_hashes = set()
segments_lock = multiprocessing.Lock()
segments_being_compared = set()


def init_comparisons_logger():
    global logger
    logger = easy_tf_log.Logger()
    logger.set_log_dir(web_globals._segments_dir)


def get_segment(hash):
    with open(os.path.join(web_globals._segments_dir, hash + '.pkl'), 'rb') as f:
        rollout = pickle.load(f)
    return rollout


def log_pct_labelled():
    global labelled_seg_hashes
    with open(os.path.join(web_globals._segments_dir, 'all_segment_hashes.txt'), 'r') as f:
        all_seg_hashes = set(f.read().strip().split('\n'))
    # Check that all labelled_seg_hashes are in all_seg_hashes
    # I.e. that all_seg_hashes is a superset of labelled_seg_hashes
    assert len(labelled_seg_hashes - all_seg_hashes) == 0
    pct_labelled = 100 * len(labelled_seg_hashes) / len(all_seg_hashes)
    logger.logkv('pct_segs_labelled', pct_labelled)


def mark_compared(hash1, hash2, preferred):
    with open(os.path.join(web_globals._segments_dir, 'compared_segments.txt'), 'a') as f:
        f.write(f'{hash1} {hash2} {preferred}\n')
    labelled_seg_hashes.add(hash1)
    labelled_seg_hashes.add(hash2)
    log_pct_labelled()


def already_compared(hash1, hash2):
    fname = os.path.join(web_globals._segments_dir, 'compared_segments.txt')
    if not os.path.exists(fname):
        open(fname, 'w').close()
        return False
    with open(fname, 'r') as f:
        lines = f.read().rstrip().split('\n')
        compared_pairs = [line.split()[:2] for line in lines]
    if [hash1, hash2] in compared_pairs or [hash2, hash1] in compared_pairs:
        return True
    else:
        return False


def sample_seg_pair():
    segment_hashes = [os.path.basename(fname).split('.')[0]
                      for fname in glob.glob(os.path.join(web_globals._segments_dir, '*.pkl'))]
    random.shuffle(segment_hashes)
    possible_pairs = combinations(segment_hashes, 2)
    for h1, h2 in possible_pairs:
        if not already_compared(h1, h2):
            return h1, h2
    raise IndexError("No segment pairs yet untested")


@comparisons_app.route('/compare_segments', methods=['GET'])
def compare_segments():
    return render_template('compare_segments.html')


@comparisons_app.route('/get_segment_video')
@nocache
def get_segment_video():
    filename = request.args['filename']
    return send_from_directory(web_globals._segments_dir, filename)


@comparisons_app.route('/get_comparison', methods=['GET'])
def get_comparison():
    global segments_lock, segments_being_compared

    if check_throttle(throttler.EventType.INTERACTION):
        return 'No segments available'

    try:
        with segments_lock:
            sampled_hashes = sample_seg_pair()
            segments_being_compared.add(sampled_hashes[0])
            segments_being_compared.add(sampled_hashes[1])
    except IndexError as e:
        msg = str(e)
        print(msg)
        return (json.dumps({}))
    segments = {}
    for hash in sampled_hashes:
        segments[hash] = get_segment(hash)

    generating_policy = None  # to match the dict from demonstrations.py, to make oracle simpler
    segment_dict = {segment_hash_str: (generating_policy, segment.vid_filename, segment.rewards)
                    for segment_hash_str, segment in segments.items()}
    return json.dumps(segment_dict)


@comparisons_app.route('/prefer_segment', methods=['POST'])
def choose_segment():
    hash1 = request.form['hash1']
    hash2 = request.form['hash2']
    pref = json.loads(request.form['pref'])
    print(hash1, hash2, pref)

    segments_being_compared.remove(hash1)
    segments_being_compared.remove(hash2)

    if pref is None:
        chosen_segment_n = 'n'
    elif pref == [0.5, 0.5]:
        add_pref(get_segment(hash1), get_segment(hash2), [0.5, 0.5])
        chosen_segment_n = 'e'
    elif pref == [1, 0]:
        chosen_segment = get_segment(hash1)
        other_segment = get_segment(hash2)
        add_pref(chosen_segment, other_segment, [1.0, 0.0])
        chosen_segment_n = '1'
    elif pref == [0, 1]:
        chosen_segment = get_segment(hash2)
        other_segment = get_segment(hash1)
        add_pref(chosen_segment, other_segment, [1.0, 0.0])
        chosen_segment_n = '2'
    else:
        return f"Error: invalid preference '{pref}'"

    mark_compared(hash1, hash2, chosen_segment_n)
    throttler.mark_event(throttler.EventType.INTERACTION)

    return ""


def prune_old_segments(segments_dir, n_to_keep):
    global segments_being_compared
    pkl_files = glob.glob(os.path.join(segments_dir, '*.pkl'))
    pkl_files.sort(key=lambda fname: os.path.getmtime(fname))
    n_to_prune = len(pkl_files) - n_to_keep
    if n_to_prune <= 0:
        return
    prune_pkl_files = pkl_files[:n_to_prune]
    prune_hashes = [str(os.path.basename(f)).split('.')[0] for f in prune_pkl_files]
    with segments_lock:
        for hash in prune_hashes:
            if hash in segments_being_compared:
                continue
            for prune_file in glob.glob(os.path.join(segments_dir, hash + '.*')):
                if global_variables.render_segments:
                    archive_segment_file(segments_dir, prune_file)
                else:
                    os.remove(prune_file)


def archive_segment_file(segments_dir, seg_file):
    old_segs_dir = os.path.join(segments_dir, 'old_segs')
    os.makedirs(old_segs_dir, exist_ok=True)
    # It's unlikely we'll get two segments with the same ID, but not impossible
    dst = os.path.join(old_segs_dir, os.path.basename(seg_file))
    if os.path.exists(dst):
        os.remove(dst)
    shutil.move(seg_file, dst)


def monitor_segments_dir_loop(dir, n_to_keep):
    global segments_being_compared
    logger = easy_tf_log.Logger()
    logger.set_log_dir(dir)
    while True:
        time.sleep(5)
        prune_old_segments(dir, n_to_keep)
        n_segments = len(glob.glob(os.path.join(dir, '*.pkl')))
        logger.logkv('episode_segments/n_segments', n_segments)
        logger.logkv('episode_segments/n_segments_being_compared', len(segments_being_compared))


def write_segments_loop(queue, dir):
    while True:
        obses, rewards, frames = queue.get()
        frames = make_small_change(frames)
        segment = CompressedRollout(final_env_state=None,
                                    obses=obses,
                                    rewards=rewards,
                                    frames=frames,
                                    vid_filename=None,
                                    generating_policy=None,
                                    actions=None)
        base_name = os.path.join(dir, str(segment.hash))
        vid_filename = base_name + '.mp4'
        save_video(segment.frames, vid_filename)
        segment.vid_filename = os.path.basename(vid_filename)
        with open(base_name + '.pkl', 'wb') as f:
            pickle.dump(segment, f)
        # Needs to be atomic because it's read asynchronously by comparisons.py
        p = os.path.join(dir, 'all_segment_hashes.txt')
        if os.path.exists(p):
            with open(p, 'r') as f:
                all_segments = f.read()
        else:
            all_segments = ''
        with atomic_write.atomic_write(p) as f:
            f.write(all_segments + str(segment.hash) + '\n')
