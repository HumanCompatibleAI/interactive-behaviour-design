#!/usr/bin/env python

"""
Start a single training run with control performed by an oracle.
"""

import argparse
import glob
import os
import random
import socket
import subprocess
import sys
import time
import uuid
from threading import Thread

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import save_args, get_git_rev, read_events_file, split_preserving_seps
import drlhp.training
from drlhp.training import FileBasedEventPipe

global_args = None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_id')
    parser.add_argument('training_mode')
    parser.add_argument('segment_generation',
                        choices=['demonstrations', 'drlhp', 'demonstrations-drlhp', 'sdrlhpnp', 'sdrlhpnp-drlhp', 'none'])
    parser.add_argument('run_name')
    parser.add_argument('--n_envs', type=int, default=16)
    parser.add_argument('--n_initial_prefs', type=int, default=500)
    parser.add_argument('--n_initial_demos', type=int, default=10)
    parser.add_argument('--pretrain_reward_predictor_epochs', type=int, default=200)
    parser.add_argument('--tmux_sess')
    default_log_dir = 'runs'
    parser.add_argument('--log_dir', default=default_log_dir)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--disable_redo', action='store_true')
    parser.add_argument('--extra_args')
    parser.add_argument('--time', default=str(int(time.time())))
    parser.add_argument('--min_label_interval_seconds', type=float, default=0.1)
    parser.add_argument('--max_interactions', type=int, default=None)
    parser.add_argument('--decay_label_rate', action='store_true')
    parser.add_argument('--gpus', default='')
    parser.add_argument('--no_pretrain', action='store_true')
    parser.add_argument('--port', type=int, default=-1)
    parser.add_argument('--tags')
    parser.add_argument('--just_pretrain', action='store_true')
    parser.add_argument('--train_using_reward_predictor_checkpoint', action='store_true')
    parser.add_argument('--reward_predictor_checkpoint')
    args = parser.parse_args()

    if args.reward_predictor_checkpoint is not None and not args.train_using_reward_predictor_checkpoint:
        raise argparse.ArgumentError("--reward_predictor_checkpoint should only be specified with "
                                     "--train_using_reward_predictor_checkpoint")
    if args.train_using_reward_predictor_checkpoint and args.reward_predictor_checkpoint is None:
        raise argparse.ArgumentError("--reward_predictor_checkpoint must be specified for "
                                     "--train_using_reward_predictor_checkpoint")
    if args.train_using_reward_predictor_checkpoint and args.training_mode != 'reward_only':
        raise argparse.ArgumentError("training_mode must reward_only for --train_using_reward_predictor_checkpoint")

    args.run_name += "_" + args.time

    git_rev = get_git_rev()
    log_dir = os.path.join(args.log_dir, f'{args.run_name}_{git_rev}')
    log_dir = os.path.abspath(log_dir)
    args.log_dir = log_dir

    return args


def main():
    global global_args

    stagger = random.randint(1, 20)
    time.sleep(stagger)

    args = get_args()
    global_args = args
    os.makedirs(args.log_dir, exist_ok=True)

    if not args.tmux_sess:
        script_name = sys.argv[0]
        script_args = '"' + '" "'.join(sys.argv[1:]) + '"'
        cmd = f"python -u {script_name} {script_args} --tmux_sess {args.run_name} --time {args.time}"
        cmd += f' 2>&1 | tee {args.log_dir}/auto_train.log'
        start_tmux_sess_with_cmd(sess_name=args.run_name, cmd=cmd, gpus=args.gpus)
        return

    save_args(args, args.log_dir, 'auto_train_args.txt')
    if args.port < 0:
        port = get_open_port()
    else:
        port = args.port
    print('launching on port: {}'.format(port))
    base_url = f'http://localhost:{port}'

    train_window_name = start_app(base_url, args.env_id, args.n_envs, port, args.seed, args.log_dir, args.tmux_sess,
                                  args.disable_redo,
                                  args.extra_args, args.segment_generation, args.gpus)

    if args.train_using_reward_predictor_checkpoint:
        add_master_policy(base_url)
        load_reward_predictor_checkpoint(base_url, args.reward_predictor_checkpoint)
    elif args.segment_generation == 'drlhp':
        # In DRLHP mode, the master policy itself generates segments, so it needs to be added right at the beginning
        add_master_policy(base_url)
        wait_for_drlhp_segments(base_url)
        oracle_window_name = start_oracle(base_url, args.segment_generation, args.tmux_sess, args.log_dir,
                                          args.min_label_interval_seconds, args.decay_label_rate)
        start_kill_oracle_after_n_interactions_thread(args.max_interactions, args.log_dir, oracle_window_name)
        if not args.no_pretrain:
            wait_for_initial_preferences(base_url, args.n_initial_prefs)
        start_reward_predictor_training(base_url)
        if not args.no_pretrain:
            print("Pretraining reward predictor...")
            wait_for_reward_predictor_n_epochs_trained(args.log_dir, args.pretrain_reward_predictor_epochs)
    elif args.segment_generation == 'demonstrations':
        # In demonstrations mode, segments are generated by subpolicies, so we don't add the master policy yet
        wait_for_demonstration_rollouts(base_url)
        oracle_window_name = start_oracle(base_url, args.segment_generation, args.tmux_sess, args.log_dir,
                                          args.min_label_interval_seconds, args.decay_label_rate)
        start_kill_oracle_after_n_interactions_thread(args.max_interactions, args.log_dir, oracle_window_name)
        if not args.no_pretrain:
            wait_for_initial_demonstrations(base_url, args.n_initial_demos)
        if 'reward' in args.training_mode:
            # We assume we already have sufficient initial preferences from initial demonstrations
            start_reward_predictor_training(base_url)
            if not args.no_pretrain:
                print("Pretraining reward predictor...")
                wait_for_reward_predictor_n_epochs_trained(args.log_dir, args.pretrain_reward_predictor_epochs)
        add_master_policy(base_url)
    elif args.segment_generation in ['demonstrations-drlhp', 'sdrlhpnp-drlhp']:
        add_master_policy(base_url)
        wait_for_demonstration_rollouts(base_url)
        wait_for_drlhp_segments(base_url)
        start_oracle(base_url, 'both', args.tmux_sess, args.log_dir, args.min_label_interval_seconds,
                     args.decay_label_rate)
        if not args.no_pretrain:
            wait_for_initial_demonstrations(base_url, args.n_initial_demos)
        start_reward_predictor_training(base_url)
        if not args.no_pretrain:
            print("Pretraining reward predictor...")
            wait_for_reward_predictor_n_epochs_trained(args.log_dir, args.pretrain_reward_predictor_epochs)
    elif args.segment_generation == 'sdrlhpnp':
        # The master policy itself also generates segments here
        add_master_policy(base_url)
        wait_for_demonstration_rollouts(base_url)
        oracle_window_name = start_oracle(base_url, args.segment_generation, args.tmux_sess, args.log_dir,
                                          args.min_label_interval_seconds, args.decay_label_rate)
        start_kill_oracle_after_n_interactions_thread(args.max_interactions, args.log_dir, oracle_window_name)
        if not args.no_pretrain:
            wait_for_initial_preferences(base_url, args.n_initial_prefs)
        if 'reward' in args.training_mode:
            start_reward_predictor_training(base_url)
            if not args.no_pretrain:
                print("Pretraining reward predictor...")
                wait_for_reward_predictor_n_epochs_trained(args.log_dir, args.pretrain_reward_predictor_epochs)
            if args.just_pretrain:
                for name in [train_window_name, oracle_window_name]:
                    cmd = ['tmux', 'kill-window', '-t', name]
                    subprocess.run(cmd)
                exit()

    else:
        raise Exception()
    # configure_env_resets(base_url)
    start_training(base_url, args.training_mode)


def load_reward_predictor_checkpoint(base_url, ckpt_path):
    requests.get(base_url + f'/run_cmd?cmd=load_rp_ckpt_path&path={ckpt_path}').raise_for_status()


def wait_for_reward_predictor_n_epochs_trained(log_dir, n):
    if n == 0:
        return
    n_epochs = get_reward_predictor_n_epochs_trained(log_dir)
    last_n_epochs = n_epochs
    while n_epochs < n:
        time.sleep(1)
        n_epochs = get_reward_predictor_n_epochs_trained(log_dir)
        if n_epochs != last_n_epochs:
            print(f"{n_epochs}/{n} epochs")
        last_n_epochs = n_epochs
    force_reward_predictor_training_process_save(global_args.log_dir)
    force_reward_predictor_load(global_args.log_dir)


def get_reward_predictor_n_epochs_trained(log_dir):
    events_file_glob = glob.glob(os.path.join(log_dir, 'reward_predictor_training', 'misc', 'events*'))
    try:
        events_file = events_file_glob[0]
    except IndexError:
        raise Exception("Couldn't find reward predictor training events file")
    events = read_events_file(events_file)
    try:
        n_epochs_timestamp_value_tuples = events['reward_predictor/n_epochs']
    except KeyError:
        raise Exception("Couldn't find no. epochs in reward predictor training events")
    return int(n_epochs_timestamp_value_tuples[-1][1])


def start_kill_oracle_after_n_interactions_thread(n, log_dir, oracle_window_name):
    if n is None:
        return

    def f():
        wait_for_n_interactions(log_dir, n)
        print(f"{n} interactions finished; killing oracle")
        cmd = ['tmux', 'kill-window', '-t', oracle_window_name]
        subprocess.run(cmd)

    Thread(target=f).start()


def start_app(base_url, env_id, n_envs, port, seed, log_dir, tmux_sess, disable_redo, extra_args, segment_generation,
              gpus):
    cmd = f'python -u run.py {env_id} --n_envs {n_envs} --port {port} --log_dir {log_dir} --seed {seed}'
    if segment_generation in ['sdrlhpnp', 'sdrlhpnp-drlhp']:
        cmd += ' --rollout_mode cur_policy'
    else:
        if not disable_redo:
            cmd += ' --redo_policy'
        if 'Seaquest' in env_id:
            cmd += ' --load_policy_ckpt_dir subpolicies/seaquest'
        elif 'LunarLander' in env_id:
            cmd += ' --load_policy_ckpt_dir subpolicies/lunarlander'
        elif 'Fetch' in env_id:
            cmd += ' --add_manual_fetch_policies'
    if extra_args is not None:
        cmd += ' ' + extra_args
    cmd += f' 2>&1 | tee {log_dir}/output.log'
    window_name = run_in_tmux_sess(tmux_sess, cmd, "app", gpus=gpus)
    print("Waiting for app to start...")
    while True:
        try:
            requests.get(base_url + '/get_status')
        except:
            time.sleep(0.5)
        else:
            break
    return window_name


def add_master_policy(base_url):
    print("Adding master policy...")
    requests.get(base_url + '/run_cmd?cmd=add_policy&name=master').raise_for_status()
    while True:
        time.sleep(0.5)
        response = requests.get(base_url + '/get_status').json()
        if 'master' in response['Policies']:
            break


def wait_for_demonstration_rollouts(base_url):
    print("Waiting for demonstration rollouts...")
    requests.get(base_url + '/generate_rollouts?policies=').raise_for_status()
    check_url = base_url + '/get_rollouts'
    while True:
        response = requests.get(check_url).json()
        if response:
            break
        else:
            time.sleep(0.5)


def wait_for_drlhp_segments(base_url):
    requests.get(base_url + '/run_cmd?cmd=use_policy&name=master').raise_for_status()
    requests.get(base_url + '/run_cmd?cmd=training_mode&mode=no_training').raise_for_status()
    print("Waiting for segments...")
    check_url = base_url + '/get_comparison'
    while True:
        response = requests.get(check_url).json()
        if response:
            break
        else:
            time.sleep(0.5)


def start_oracle(base_url, segment_generation, tmux_sess, log_dir, min_label_interval_seconds, decay_label_rate):
    if segment_generation == 'sdrlhpnp':
        segment_generation = 'demonstrations'
    decay_arg = '--decay_label_rate' if decay_label_rate else ''
    cmd = (
        f'python -u oracle.py {base_url} {segment_generation} '
        f'--seconds_per_label {min_label_interval_seconds} --log_dir {log_dir} {decay_arg}'
        f' 2>&1 | tee {log_dir}/oracle.log')
    oracle_window_name = run_in_tmux_sess(tmux_sess, cmd, "oracle", gpus='')
    return oracle_window_name


def wait_for_n_interactions(log_dir, n):
    search_string = f"Simulated {n} interactions"
    while True:
        try:
            oracle_log = open(os.path.join(log_dir, 'oracle.log'), 'r').read()
        except Exception as e:
            print("While reading oracle log:", e)
        else:
            if search_string in oracle_log:
                return True
        time.sleep(1)


def wait_for_initial_demonstrations(base_url, n_initial_demos):
    n_demos = 0
    while n_demos < n_initial_demos:
        print(f"Waiting for initial demonstrations ({n_demos} clips so far)")
        time.sleep(1)
        n_demos = get_n_demos(base_url)


def wait_for_initial_preferences(base_url, n_initial_prefs):
    n_prefs = 0
    while n_prefs < n_initial_prefs:
        print(f"Waiting for initial preferences ({n_prefs} so far)")
        time.sleep(1)
        n_prefs = get_n_prefs(base_url)
    # So that preferences are picked up by reward predictor training process
    force_main_process_checkpoint(base_url)


def force_main_process_checkpoint(base_url):
    requests.get(base_url + '/run_cmd?cmd=checkpoint').raise_for_status()


def force_reward_predictor_training_process_save(log_dir):
    FileBasedEventPipe.send_event(os.path.join(log_dir, drlhp.training.FORCE_SAVE_FNAME))
    FileBasedEventPipe.wait_for_ack(os.path.join(log_dir, drlhp.training.FORCE_SAVE_FNAME))


def force_reward_predictor_load(log_dir):
    FileBasedEventPipe.send_event(os.path.join(log_dir, drlhp.training.FORCE_LOAD_FNAME))
    FileBasedEventPipe.wait_for_ack(os.path.join(log_dir, drlhp.training.FORCE_LOAD_FNAME))


def start_reward_predictor_training(base_url):
    requests.get(base_url + '/run_cmd?cmd=start_drlhp_training').raise_for_status()


def start_training(base_url, training_mode):
    print("Starting training...")
    requests.get(base_url + '/run_cmd?cmd=config_demos&noop_actions=False').raise_for_status()
    requests.get(base_url + '/run_cmd?cmd=use_policy&name=master').raise_for_status()
    requests.get(base_url + f'/run_cmd?cmd=training_mode&mode={training_mode}')
    if 'reward' in training_mode:
        requests.get(base_url + '/run_cmd?cmd=set_reward_source&src=drlhp').raise_for_status()


def configure_env_resets(base_url):
    requests.get(base_url + '/run_cmd?cmd=add_reset_pool&name=random_states_from_episode&max_len=100')
    requests.get(base_url + '/run_cmd?cmd=use_reset_pool&from=training&name=random_states_from_episode')
    requests.get(base_url + '/run_cmd?cmd=use_reset_pool&to=demonstrations&name=random_states_from_episode')
    requests.get(base_url + '/run_cmd?cmd=set_demonstrations_reset_mode&mode=from_state_cache')


# Helper functions

def get_n_prefs(base_url):
    return int(requests.get(base_url + '/get_status').json()['No. prefs'])


def get_n_demos(base_url):
    return int(requests.get(base_url + '/get_status').json()['No. demonstrated episodes'])


def start_tmux_sess_with_cmd(sess_name, cmd, gpus):
    global global_args
    cmd = f'CUDA_VISIBLE_DEVICES="{gpus}" ' + cmd
    cmd += '; echo; read -p "Press enter to exit..."'
    cmd = ['tmux', 'new-sess', '-d', '-s', sess_name, '-n', f'{sess_name}-main', cmd]
    subprocess.run(cmd)


def run_in_tmux_sess(sess_name, cmd, window_name, gpus):
    global global_args
    window_name += '_' + str(uuid.uuid4())[:4]

    cmd = f'CUDA_VISIBLE_DEVICES="{gpus}" ' + cmd
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


if __name__ == '__main__':
    main()
