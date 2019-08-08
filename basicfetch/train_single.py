import argparse
import multiprocessing
import operator
import os
import random
import subprocess
import sys
from functools import reduce

import gym
from gym.envs.registration import register
from gym.wrappers import Monitor

import baselines
import basicfetch
from baselines import logger
from baselines.run import main as baselines_run_main
from reward_functions import reward_function_dict

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    if random.random() < 0.5:
        d = '0'
    else:
        d = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = d


def get_git_rev():
    try:
        cmd = 'git rev-parse --short HEAD'
        git_rev = subprocess.check_output(cmd.split(' '), stderr=subprocess.PIPE).decode().rstrip()
        return git_rev
    except subprocess.CalledProcessError:
        return 'unkrev'


parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('reward_spec')
parser.add_argument('--n_envs', type=int, default=16)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--ckpt')
args = parser.parse_args()
args.dir += '_' + get_git_rev()
os.environ["OPENAI_LOGDIR"] = args.dir
os.environ["OPENAI_LOG_FORMAT"] = 'stdout,log,csv,tensorboard'

basicfetch.register()
first_env_semaphore = multiprocessing.Semaphore()


def get_reward_function(name):
    reward_function = reward_function_dict
    for t in name.split('.'):
        reward_function = reward_function[t]
    assert callable(reward_function)
    return reward_function


def construct_reward_function():
    funcs = []
    for name in args.reward_spec.split(' '):
        funcs.append(get_reward_function(name))

    def f(quat, pos):
        return reduce(operator.mul, [f(quat, pos) for f in funcs])

    return f


def make_env():
    env = gym.make(f'FetchBasicDelta-v0')
    env.unwrapped.reward_func = construct_reward_function()
    if first_env_semaphore.acquire(timeout=0):
        env = Monitor(env, video_callable=lambda n: n % 5 == 0, directory=logger.get_dir())
    return env


register(
    id='E-v0',
    entry_point=make_env,
)

baselines.run._game_envs['robotics'].add('E-v0')
# 1e5 total timesteps on 16 workers is about 5 minutes
# 1e6 is about an hour
arg_str = f"--alg=ppo2 --env=E-v0 --num_env {args.n_envs} --num_timesteps 1e6 --seed {args.seed} "
arg_str += f"--save_path {os.path.join(args.dir, 'saved_model')} --log_interval 3 --save_interval 10"
arg_str += f" --load_path {args.ckpt}"
sys.argv = arg_str.split(" ")
baselines_run_main()
