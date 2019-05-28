#!/usr/bin/env python


import argparse
import os
import socket
import subprocess
import sys
import time
import uuid

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from wrappers.fetch_pick_and_place_register import register
from utils import save_args

register()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_id')
    parser.add_argument('run_name')
    parser.add_argument('log_dir')
    parser.add_argument('--n_demos', type=int)
    parser.add_argument('--tmux_sess')
    parser.add_argument('--time', default=str(int(time.time())))
    args = parser.parse_args()

    args.run_name += "_" + args.time
    git_rev = get_git_rev()
    log_dir = os.path.join(args.log_dir, f'{args.run_name}_{git_rev}')
    log_dir = os.path.abspath(log_dir)
    args.log_dir = log_dir

    return args


def main():
    args = get_args()
    os.makedirs(args.log_dir, exist_ok=True)

    if not args.tmux_sess:
        script_name = sys.argv[0]
        script_args = '"' + '" "'.join(sys.argv[1:]) + '"'
        cmd = f'python -u {script_name} {script_args} --tmux_sess {args.run_name} --time {args.time}'
        cmd += f' 2>&1 | tee {args.log_dir}/auto_train.log'
        start_tmux_sess_with_cmd(sess_name=args.run_name, cmd=cmd)
        return

    save_args(args, args.log_dir, 'auto_train_args.txt')
    port = get_open_port()
    base_url = f'http://localhost:{port}'

    start_app(base_url, args.env_id, port, args.log_dir, args.tmux_sess)

    wait_for_demonstration_rollouts(base_url)
    start_oracle(base_url, args.tmux_sess, args.log_dir)


def start_app(base_url, env_id, port, log_dir, tmux_sess):
    if 'Fetch' in env_id:
        disable_redo = True
        if 'Repeat1' in env_id:
            rollout_len_seconds = 0.5
        elif 'Repeat3' in env_id:
            rollout_len_seconds = 0.15
        else:
            raise Exception()
    else:
        raise Exception()
    cmd = f'python -u run.py {env_id} --n_envs 1 --port {port} --log_dir {log_dir} --rollout_length_seconds {rollout_len_seconds}'
    if not disable_redo:
        cmd += ' --redo_policy'
    if 'Seaquest' in env_id:
        cmd += ' --load_policy_ckpt_dir subpolicies/seaquest'
    elif 'LunarLander' in env_id:
        cmd += ' --load_policy_ckpt_dir subpolicies/lunarlander'
    elif 'Fetch' in env_id:
        cmd += ' --add_manual_fetch_policies'
    cmd += f' 2>&1 | tee {log_dir}/output.log'
    run_in_tmux_sess(tmux_sess, cmd, "app")
    print("Waiting for app to start...")
    while True:
        try:
            requests.get(base_url + '/get_status')
        except:
            time.sleep(0.5)
        else:
            break


def wait_for_demonstration_rollouts(base_url):
    requests.get(base_url + '/generate_rollouts?policies=').raise_for_status()
    print("Waiting for demonstration rollouts...")
    check_url = base_url + '/get_rollouts'
    while True:
        response = requests.get(check_url).json()
        if response:
            break
        else:
            time.sleep(0.5)


def start_oracle(base_url, tmux_sess, log_dir):
    cmd = (f'python -u oracle.py {base_url} demonstrations 0'
           f' 2>&1 | tee {log_dir}/oracle.log')
    oracle_window_name = run_in_tmux_sess(tmux_sess, cmd, "oracle")
    return oracle_window_name


# Helper functions


def start_tmux_sess_with_cmd(sess_name, cmd):
    cmd += '; echo; read -p "Press enter to exit..."'
    cmd = ['tmux', 'new-sess', '-d', '-s', sess_name, '-n', f'{sess_name}-main', cmd]
    subprocess.run(cmd)


def run_in_tmux_sess(sess_name, cmd, window_name):
    window_name += '_' + str(uuid.uuid4())[:4]
    cmd += '; echo; read -p "Press enter to exit..."'
    tmux_cmd = ['tmux', 'new-window', '-ad', '-t', f'{sess_name}-main', '-n', window_name, cmd]
    subprocess.run(tmux_cmd)
    return window_name


def get_open_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def get_git_rev():
    try:
        cmd = 'git rev-parse --short HEAD'
        git_rev = subprocess.check_output(cmd.split(' '), stderr=subprocess.PIPE).decode().rstrip()
        return git_rev
    except subprocess.CalledProcessError:
        return 'unkrev'


if __name__ == '__main__':
    main()
