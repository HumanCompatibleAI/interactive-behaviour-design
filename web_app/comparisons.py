import glob
import json
import os
import pickle
import random
from itertools import combinations

import easy_tf_log
from flask import Blueprint, render_template, request, send_from_directory

import global_variables
from web_app import web_globals
from web_app.utils import add_pref, nocache, get_n_rl_steps

comparisons_app = Blueprint('comparisons', __name__)
logger = easy_tf_log.Logger()
logger.set_log_dir(web_globals._segments_dir)
n_rl_steps_at_last_pref = None
labelled_seg_hashes = set()


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
    global n_rl_steps_at_last_pref
    n_rl_steps = get_n_rl_steps()
    if n_rl_steps is not None and n_rl_steps_at_last_pref is not None:  # Maybe we haven't started training yet
        n_rl_steps_since_last_pref = n_rl_steps - n_rl_steps_at_last_pref
        if n_rl_steps_since_last_pref < global_variables.min_n_rl_steps_per_pref:
            return 'No segments available'

    try:
        sampled_hashes = sample_seg_pair()
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
    global n_rl_steps_at_last_pref

    hash1 = request.form['hash1']
    hash2 = request.form['hash2']
    pref = json.loads(request.form['pref'])
    print(hash1, hash2, pref)

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

    n_rl_steps_at_last_pref = get_n_rl_steps()
    if n_rl_steps_at_last_pref is not None:
        logger.logkv('comparisons/n_rl_steps_at_last_pref', n_rl_steps_at_last_pref)

    return ""
