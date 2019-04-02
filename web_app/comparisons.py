import glob
import json
import os
import pickle
import random
from itertools import combinations

from web_app import web_globals
from web_app.utils import add_pref, nocache

from flask import Blueprint, render_template, request, send_from_directory

comparisons_app = Blueprint('comparisons', __name__)


def get_segment(hash):
    with open(os.path.join(web_globals._segments_dir, hash + '.pkl'), 'rb') as f:
        rollout = pickle.load(f)
    return rollout


def mark_compared(hash1, hash2, preferred):
    with open(os.path.join(web_globals._segments_dir, 'compared_segments.txt'), 'a') as f:
        f.write(f'{hash1} {hash2} {preferred}\n')


def already_compared(hash1, hash2):
    fname = os.path.join(web_globals._segments_dir, 'compared_segments.txt')
    if not os.path.exists(fname):
        open(fname, 'w').close()
        return False
    with open(fname, 'r') as f:
        lines = f.read().rstrip().split('\n')
        compared_pairs = [line.split() for line in lines]
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
    try:
        sampled_hashes = sample_seg_pair()
    except IndexError as e:
        msg = str(e)
        print(msg)
        return(json.dumps({}))
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

    return ""
