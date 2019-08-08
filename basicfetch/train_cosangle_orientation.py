#!/usr/bin/env python3

import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('runs_dir')
args = parser.parse_args()


def start_tmux_sess_with_cmd(sess_name, cmd):
    cmd += '; echo; read -p "Press enter to exit..."'
    cmd = ['tmux', 'new-sess', '-d', '-s', sess_name, '-n', f'{sess_name}-main', cmd]
    subprocess.run(cmd)


def run_in_tmux_sess(sess_name, cmd, window_name):
    cmd += '; echo; read -p "Press enter to exit..."'
    tmux_cmd = ['tmux', 'new-window', '-ad', '-t', f'{sess_name}-main', '-n', window_name, cmd]
    subprocess.run(tmux_cmd)


reward_types = [
    'cosangle.right orientation',
    'cosangle.left orientation',
    'cosangle.up orientation',
    'cosangle.down orientation',
    'cosangle.forward orientation',
    'cosangle.backward orientation',
]

start_tmux_sess_with_cmd('fetchbasic', 'echo "Dummy window"')
for seed in [0, 1, 2]:
    for reward_type in reward_types:
        run_name = 'FetchBasic-' + reward_type.replace(' ', '-') + str(seed)
        run_dir = os.path.join(args.runs_dir, run_name)
        cmd = f"python train_single.py '{run_dir}' '{reward_type}' --seed {seed}"
        run_in_tmux_sess('fetchbasic', cmd, reward_type + str(seed))
