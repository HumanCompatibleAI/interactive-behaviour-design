import argparse
import sys

import baselines
import basicfetch
from baselines.run import main as baselines_run_main

basicfetch.register()

parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()

baselines.run._game_envs['robotics'].add('FetchBasic-v0')
arg_str = f"--alg=ppo2 --env=FetchBasic-v0 --num_env 1 --num_timesteps 0 --load_path {args.model} --play"
sys.argv = arg_str.split(" ")
baselines_run_main()
