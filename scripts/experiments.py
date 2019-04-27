import os
import sys
import argparse
import subprocess
import time
import socket
import requests
from threading import Thread

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import save_args, get_git_rev

################    Begin code   ################
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_id') #e.g. "LunarLanderStatefulStats-v0"
    parser.add_argument('run_name') #e.g. "lunarlander-0-bc"
    parser.add_argument('--n_envs', type=int, default=16)
    parser.add_argument('-t', '--training_modes', nargs='+', default=['bc'])
    parser.add_argument('--demos_path')
    parser.add_argument('--sdrlhp_prefs')
    parser.add_argument('--drlhp_prefs')
    parser.add_argument('--tmux_sess')

    if os.path.exists('/efs'):
        default_log_dir = '/efs'
    else:
        default_log_dir = 'runs'
    parser.add_argument('--log_dir', default=default_log_dir)

    parser.add_argument('--extra_args')
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
        cmd = f'python -u {script_name} {script_args} --tmux_sess {args.run_name} --time {args.time}' #run this script in tmux
        cmd += f' 2>&1 | tee {args.log_dir}/auto_experiments.log'
        start_tmux_sess_with_cmd(sess_name=args.run_name, cmd=cmd)
        return

    save_args(args, args.log_dir, 'auto_train_args.txt')

    for train_mode in args.training_modes:
        port = 5000 #get_open_port()
        base_url = f'http://localhost:{port}'
        if train_mode == 'bc':
            log_dir = args.log_dir + "/bc"
            if args.demos_path is not None and os.path.exists(args.demos_path):
                start_bc(args.env_id, args.demos_path, log_dir, port, base_url) #starts the run.py app
                add_master_policy(base_url)
                train_bc(base_url) #starts bc training commands
            else:
                print(f"Provide valid --demos_path to expert demos for behavior cloning. Path '{args.demos_path}' invalid.")
        elif train_mode == 'drlhp' or train_mode == 'sdrlhp':
            loaded_prefs_path = args.drlhp_prefs
            pref_type = 'drlhp'
            pref_generation = 'drlhp'
            if train_mode == 'sdrlhp':
                loaded_prefs_path = args.sdrlhp_prefs
                pref_type = 'sdrlhp'
                pref_generation = 'demonstrations'

            if loaded_prefs_path is not None and os.path.exists(loaded_prefs_path): #train from loaded prefs
                start_rl_from_prefs(args.env_id, loaded_prefs_path, pref_type, args.log_dir, port, base_url)
                add_master_policy(base_url)
                pretrain_sec = 5 * 60 # 5 min
                start_reward_predictor_training(base_url, pretrain_sec)
                train_rl(base_url)
            else: #train using oracle prefs
                print(f"Starting {pref_type} training from oracle preferences.")
                start_oracle_training(args.env_id, pref_generation, pref_type, args.log_dir)
        elif train_mode == 'bc+drlhp' or train_mode == 'bc+sdrlhp':
            print(f"Starting {train_mode} training...")
            log_dir = args.log_dir + '/' + train_mode
            #pretrain with bc
            if args.demos_path is not None and os.path.exists(args.demos_path):
                start_bc(args.env_id, args.demos_path, log_dir, port, base_url) #starts the run.py app
                add_master_policy(base_url)
                bc_train_sec = 5 * 60 #5 min
                train_bc(base_url, bc_train_sec) #train bc for train_sec
            else:
                print(f"Provide valid --demos_path to expert demos for behavior cloning. Path '{args.demos_path}' invalid.")
                continue

            #then train with (s)drlhp (oracle)
            min_label_interval_seconds = 3
            pretrain_reward_predictor_sec = 5 * 60 # 5 min
            n_initial_prefs = 500
            n_initial_demos = 10
            max_interactions = n_initial_demos * 20
            pref_generation = 'drlhp'
            if train_mode == 'bc+sdrlhp':
                pref_generation = 'demonstrations'
                wait_for_demonstration_rollouts(base_url)

            oracle_window_name = start_oracle(base_url, pref_generation, log_dir,
                                              min_label_interval_seconds)
            start_kill_oracle_after_n_interactions_thread(max_interactions, log_dir, oracle_window_name)
            if train_mode == 'bc+drlhp':
                wait_for_initial_preferences(base_url, n_initial_prefs)
            else:
                wait_for_initial_demonstrations(base_url, n_initial_demos)
            start_reward_predictor_training(base_url, pretrain_reward_predictor_sec)
            train_rl(base_url)


def start_bc(env, expert_demos_path, log_dir, port, base_url):
    print(f"Running behavioral cloning (port {port}) on demos in {expert_demos_path}")
    cmd = f'python -u run.py {env} --load_sdrlhp_demos {expert_demos_path} --port {port} --log_dir {log_dir}'

    if 'Seaquest' in env:
        cmd += ' --load_policy_ckpt_dir subpolicies/seaquest'
    elif 'LunarLander' in env:
        cmd += ' --load_policy_ckpt_dir subpolicies/lunarlander'
    elif 'Fetch' in env:
        cmd += ' --add_manual_fetch_policies'

    run_in_tmux_sess(cmd, 'bc')
    print("Waiting for bc app to start...")
    while True:
        try:
            requests.get(base_url + '/get_status')
        except:
            time.sleep(0.5)
        else:
            break

def start_rl_from_prefs(env, prefs_path, pref_type, log_dir, port, base_url):
    assert pref_type in ['drlhp', 'sdrlhp']

    print(f"Running rl (port {port}) on preferences in {prefs_path}")
    log_dir += f"/{pref_type}"
    cmd = f'python -u run.py {env} --load_{pref_type}_prefs {prefs_path} --render_segments --port {port} --log_dir {log_dir}'

    if 'Seaquest' in env:
        cmd += ' --load_policy_ckpt_dir subpolicies/seaquest'
    elif 'LunarLander' in env:
        cmd += ' --load_policy_ckpt_dir subpolicies/lunarlander'
    elif 'Fetch' in env:
        cmd += ' --add_manual_fetch_policies'

    run_in_tmux_sess(cmd, pref_type)
    print(f"Waiting for {pref_type} app to start...")
    while True:
        try:
            requests.get(base_url + '/get_status')
        except:
            time.sleep(0.5)
        else:
            break

def start_oracle(base_url, segment_generation, log_dir, min_label_interval_seconds):
    cmd = (f'python -u oracle.py {base_url} {segment_generation} {min_label_interval_seconds}'
           f' 2>&1 | tee {log_dir}/oracle.log')
    oracle_window_name = run_in_tmux_sess(cmd, "oracle")
    print(f"Started oracle for {segment_generation} preferences in tmux window {oracle_window_name}")
    return oracle_window_name

def start_kill_oracle_after_n_interactions_thread(n, log_dir, oracle_window_name):
    if n is None:
        return

    def f():
        wait_for_n_interactions(log_dir, n)
        print(f"{n} interactions finished; killing oracle")
        cmd = ['tmux', 'kill-window', '-t', oracle_window_name]
        subprocess.run(cmd)

    Thread(target=f).start()

def start_oracle_training(env, pref_generation, run_name, log_dir):
    tmux_sess_name = "oracle_" + run_name
    cmd = f'python -u scripts/train/auto_train_prefs.py {env} ' \
          f'reward_only {pref_generation} {tmux_sess_name} --log_dir {log_dir}'
    cmd += f'; echo "Oracle training started in tmux sess {tmux_sess_name}"'
    run_in_tmux_sess(cmd, run_name)

def add_master_policy(base_url):
    print("Adding master policy...")
    requests.get(base_url + '/run_cmd?cmd=add_policy&name=master').raise_for_status()
    while True:
        time.sleep(0.5)
        response = requests.get(base_url + '/get_status').json()
        if 'master' in response['Policies']:
            break

def train_bc(base_url, seconds=0):
    requests.get(base_url + '/run_cmd?cmd=use_policy&name=master').raise_for_status()
    requests.get(base_url + '/run_cmd?cmd=training_mode&mode=bc_only').raise_for_status()
    print("Starting bc training...")
    check_url = base_url + '/get_comparison'
    while True:
        response = requests.get(check_url).json()
        if response:
            break
        else:
            time.sleep(0.5)

    if seconds > 0:
        print(f"Training using bc for {seconds / 60} min...")
        time.sleep(seconds)

def train_rl(base_url):
    print("Starting rl training...")
    requests.get(base_url + '/run_cmd?cmd=use_policy&name=master').raise_for_status()
    requests.get(base_url + '/run_cmd?cmd=training_mode&mode=reward_only')
    requests.get(base_url + '/run_cmd?cmd=set_reward_source&src=drlhp').raise_for_status()
    check_url = base_url + '/get_comparison'
    while True:
        response = requests.get(check_url).json()
        if response:
            break
        else:
            time.sleep(0.5)

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

def wait_for_initial_demonstrations(base_url, n_initial_demos):
    n_demos = 0
    last_n = -1
    while n_demos < n_initial_demos:
        if n_demos > last_n:
            print(f"Waiting for initial demonstrations ({n_demos} clips so far)")
            last_n = n_demos
        time.sleep(1)
        n_demos = get_n_demos(base_url)

def wait_for_initial_preferences(base_url, n_initial_prefs):
    n_prefs = 0
    last_n = -1
    while n_prefs < n_initial_prefs:
        if n_prefs > last_n:
            print(f"Waiting for initial preferences ({n_prefs} so far)")
            last_n = n_prefs
        time.sleep(1)
        n_prefs = get_n_prefs(base_url)

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

def start_reward_predictor_training(base_url, seconds):
    print(f"Pretraining reward predictor for {seconds / 60} min...")
    requests.get(base_url + '/run_cmd?cmd=start_drlhp_training').raise_for_status()
    time.sleep(seconds)

################        HELPER FUNCTIONS        ################
def get_n_prefs(base_url):
    return int(requests.get(base_url + '/get_status').json()['No. prefs'])

def get_n_demos(base_url):
    return int(requests.get(base_url + '/get_status').json()['No. demonstrated episodes'])

def start_tmux_sess_with_cmd(sess_name, cmd):
    cmd += '; echo; read -p "Press enter to exit..."'
    cmd = ['tmux', 'new-sess', '-d', '-s', sess_name, '-n', f'main', cmd]
    subprocess.run(cmd)

def run_in_tmux_sess(cmd, window_name):
    cmd += '; echo; read -p "Press enter to exit..."'
    tmux_cmd = ['tmux', 'new-window', '-ad', '-t', f'main', '-n', window_name, cmd]
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