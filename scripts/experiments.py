import os
import sys
import argparse
import subprocess
import time
import socket
import requests

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
            if args.demos_path is not None and os.path.exists(args.demos_path):
                start_bc(args.env_id, args.demos_path, args.log_dir, port, base_url)
                add_master_policy(base_url)
                train_bc(base_url)
            else:
                print(f"Provide valid path to expert demos for behavior cloning. Path '{args.demos_path}' invalid.")

def start_bc(env, expert_demos_path, log_dir, port, base_url):
    # Collect bc plots
    print(f"Running behavioral cloning (port {port}) on demos in {expert_demos_path}")
    log_dir += "/bc"
    cmd = f'python -u run.py {env} --load_sdrlhp_demos {expert_demos_path} --port {port} --log_dir {log_dir}'
    run_in_tmux_sess(cmd, 'bc')
    print("Waiting for bc app to start...")
    while True:
        try:
            requests.get(base_url + '/get_status')
        except:
            time.sleep(0.5)
        else:
            break

def get_open_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def add_master_policy(base_url):
    print("Adding master policy...")
    requests.get(base_url + '/run_cmd?cmd=add_policy&name=master').raise_for_status()
    while True:
        time.sleep(0.5)
        response = requests.get(base_url + '/get_status').json()
        if 'master' in response['Policies']:
            break

def train_bc(base_url):
    requests.get(base_url + '/run_cmd?cmd=use_policy&name=master').raise_for_status()
    requests.get(base_url + '/run_cmd?cmd=training_mode&mode=bc_only').raise_for_status()
    print("Waiting for segments...")
    check_url = base_url + '/get_comparison'
    while True:
        response = requests.get(check_url).json()
        if response:
            break
        else:
            time.sleep(0.5)

def start_tmux_sess_with_cmd(sess_name, cmd):
    cmd += '; echo; read -p "Press enter to exit..."'
    cmd = ['tmux', 'new-sess', '-d', '-s', sess_name, '-n', f'main', cmd]
    subprocess.run(cmd)

def run_in_tmux_sess(cmd, window_name):
    cmd += '; echo; read -p "Press enter to exit..."'
    tmux_cmd = ['tmux', 'new-window', '-ad', '-t', f'main', '-n', window_name, cmd]
    subprocess.run(tmux_cmd)
    return window_name

if __name__ == '__main__':
    main()