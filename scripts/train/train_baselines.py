import argparse
import multiprocessing
import os
import subprocess
import sys

import gym
from gym.envs.registration import register
from gym.wrappers import Monitor

import baselines
from baselines import logger
from baselines.run import main as baselines_run_main

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from wrappers import fetch_pick_and_place_register
from wrappers import lunar_lander_reward
from wrappers import seaquest_reward

fetch_pick_and_place_register.register()
lunar_lander_reward.register()
seaquest_reward.register()

parser = argparse.ArgumentParser()
parser.add_argument('alg')
parser.add_argument('env', help='e.g. LunarLanderStateful-v0, FetchPickAndPlaceDense1-v0')
parser.add_argument('env_type', help='e.g. box2d, robotics')
parser.add_argument('dir')
parser.add_argument('--extra_args', default='')
parser.add_argument('--n_envs', type=int, default=16)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--render_every_n_episodes', type=int, default=10)
args = parser.parse_args()


def record_context(path):
    git_info = subprocess.check_output(['git', 'status']).decode()
    git_diff = subprocess.check_output(['git', 'diff']).decode()
    git_rev = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode()
    with open(path, 'w') as f:
        f.write(git_rev)
        f.write('\n')
        f.write(git_info)
        f.write('\n')
        f.write(git_diff)
        f.write('\n')
        f.write(' '.join(sys.argv))
        f.write('\n\n')
        f.write(str(args))
        f.write('\n\n')


os.makedirs(args.dir)
record_context(os.path.join(args.dir, 'context.txt'))

os.environ["OPENAI_LOGDIR"] = args.dir
os.environ["OPENAI_LOG_FORMAT"] = 'stdout,log,csv,tensorboard'

first_env_semaphore = multiprocessing.Semaphore()


def make_env():
    env = gym.make(args.env)
    if first_env_semaphore.acquire(timeout=0):
        env = Monitor(env, video_callable=lambda n: n % args.render_every_n_episodes == 0, directory=logger.get_dir())
    return env


if args.env_type == 'atari':
    env_name = 'ENoFrameskip-v4'
else:
    env_name = 'E-v0'
register(id=env_name, entry_point=make_env, )
baselines.run._game_envs[args.env_type].add(env_name)

sys.argv = sys.argv[:1]
sys.argv.extend(f"--alg={args.alg} --env={env_name} --num_env {args.n_envs} {args.extra_args}"
                f"--num_timesteps 50e6 --seed {args.seed}".split(" "))
baselines_run_main()
