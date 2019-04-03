#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import sys

import numpy as np
from flask import Flask, render_template, request, send_from_directory

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

parser = argparse.ArgumentParser()
parser.add_argument('run_dir')
args = parser.parse_args()
segments_dir = os.path.abspath(os.path.join(args.run_dir, 'segments'))

with open(os.path.join(segments_dir, 'compared_segments.txt'), 'r') as f:
    prefs = [l.strip().split(' ') for l in f.readlines()]

script_dir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, template_folder=script_dir)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.route("/")
def main():
    return render_template('check_prefs.html')


@app.route("/get_prefs")
def get_prefs():
    return json.dumps(prefs)


@app.route("/get_vid")
def get_vid():
    hash = request.args['hash']
    filename = hash + '.mp4'
    if not os.path.exists(os.path.join(segments_dir, filename)):
        print("Warning: could not find", filename)
    return send_from_directory(segments_dir, filename)


@app.route("/check_pref")
def check_pref():
    h1 = request.args['h1']
    h2 = request.args['h2']

    with open(os.path.join(segments_dir, h1 + '.pkl'), 'rb') as f:
        r1 = pickle.load(f)
    with open(os.path.join(segments_dir, h2 + '.pkl'), 'rb') as f:
        r2 = pickle.load(f)

    r1s = np.array2string(np.array(r1.rewards), precision=1)
    r2s = np.array2string(np.array(r2.rewards), precision=1)
    info = ("Rewards:<br />"
            "Segment 1: " + r1s + " (sum {:.1f})<br />".format(sum(r1.rewards)) +
            "Segment 2: " + r2s + " (sum {:.1f})".format(sum(r2.rewards)))
    if r1.obses[0].shape == (6,):
        d1 = sum(([np.linalg.norm(o) for o in r1.obses]))
        d2 = sum(([np.linalg.norm(o) for o in r2.obses]))
        info += ("<br />"
                 "Cumulative distances from gripper to block:<br />"
                 "Segment 1: {:.3f}".format(d1) + "<br />" +
                 "Segment 2: {:.3f}".format(d2))
    else:
        info = ""

    return info


app.run()
