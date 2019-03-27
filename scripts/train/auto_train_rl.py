import os
import sys
import time
from argparse import ArgumentParser

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from auto_train_prefs import get_open_port, start_tmux_sess_with_cmd
from utils import get_git_rev


def get_args():
    parser = ArgumentParser()
    parser.add_argument('seed', type=int, default=0)
    parser.add_argument('env')
    parser.add_argument('run_name')
    if os.path.exists('/efs'):
        default_log_dir = '/efs'
    else:
        default_log_dir = 'runs'
    parser.add_argument('--log_dir', default=default_log_dir)
    return parser.parse_args()


def start_app(base_url, env, port, seed, run_name, log_dir):
    cmd = f'python -u run.py {env} --n_envs 16 --port {port} --log_dir {log_dir} --seed {seed}'
    cmd += f' 2>&1 | tee {log_dir}/output.log'
    start_tmux_sess_with_cmd(run_name, cmd)
    while True:
        try:
            requests.get(base_url + '/get_status')
        except:
            time.sleep(0.5)
        else:
            break


args = get_args()
git_rev = get_git_rev()
log_dir = os.path.abspath(os.path.join(args.log_dir, f'{args.run_name}_{git_rev}'))
os.makedirs(log_dir)
port = get_open_port()
base_url = f'http://localhost:{port}'
start_app(base_url, args.env, port, args.seed, args.run_name, log_dir)

requests.get(base_url + '/run_cmd?cmd=add_policy&name=master').raise_for_status()
while True:
    time.sleep(0.5)
    response = requests.get(base_url + '/get_status').json()
    if 'master' in response['Policies']:
        break

requests.get(base_url + '/run_cmd?cmd=set_reward_source&src=env').raise_for_status()
requests.get(base_url + '/run_cmd?cmd=use_policy&name=master').raise_for_status()
