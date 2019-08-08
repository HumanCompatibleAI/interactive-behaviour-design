#!/usr/bin/env python3

import argparse
import os
import subprocess

from reward_functions import reward_function_dict

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


reward_types = []
for reward_type, x in reward_function_dict.items():
    if reward_type == 'dummy':
        continue
    if callable(x):
        reward_types.append(reward_type)
    elif type(x) is dict:
        for reward_subtype, _ in x.items():
            reward_types.append(reward_type + '.' + reward_subtype)

start_tmux_sess_with_cmd('fetchbasic', 'echo "Dummy window"')
for seed in [0, 1, 2]:
    for reward_type in reward_types:
        run_name = 'FetchBasic-' + reward_type + str(seed)
        run_dir = os.path.join(args.runs_dir, run_name)
        cmd = f"python train_single.py '{run_dir}' '{reward_type}' --seed {seed}"
        run_in_tmux_sess('fetchbasic', cmd, reward_type + str(seed))
