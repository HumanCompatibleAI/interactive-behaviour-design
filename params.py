import logging
import time
from argparse import ArgumentParser

import os
from os import path as osp

import global_variables
from global_variables import RolloutMode, RolloutRandomness
from baselines import logger
from utils import save_args, get_git_rev


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lrschedule', help='Learning rate schedule',
                        choices=['constant', 'linear'], default='constant')
    parser.add_argument('--n_envs', type=int, default=16)
    parser.add_argument('--load_sdrlhp_demos') #runs/[run_dir]/experience/demonstration_rollouts.pkl
    parser.add_argument('--load_sdrlhp_prefs') #runs/[run_dir]/experience/pref_db.pkl
    parser.add_argument('--load_experience_dir') #runs/[run_dir]/experience/ for classifier/reset_states
    parser.add_argument('--load_classifier_ckpt')
    parser.add_argument('--load_policy_ckpt_dir')
    parser.add_argument('--load_drlhp_ckpt_dir')
    parser.add_argument('--load_drlhp_prefs')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--no_save_frames', action='store_true')
    parser.add_argument('--add_manual_fetch_policies', action='store_true')
    parser.add_argument('--render_segments', action='store_true')
    parser.add_argument('--no_render_demonstrations', action='store_true')
    parser.add_argument('--render_every_nth_episode', type=int)
    parser.add_argument('--rollout_length_seconds', type=float, default=1.0)
    parser.add_argument('--show_from_end', type=float)
    parser.add_argument('--redo_policy', action='store_true')
    parser.add_argument('--noisy_policies', action='store_true')
    parser.add_argument('--max_demonstration_length', type=int)
    parser.add_argument('--segment_save_mode', choices=['single_env', 'multi_env'], default='multi_env')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--log_dir')
    seconds_since_epoch = str(int(time.time()))
    group.add_argument('--run_name', default='test-run_{}'.format(seconds_since_epoch))
    parser.add_argument('--demonstrations_buffer_len', type=int, default=3000)
    parser.add_argument('--rstd', type=float)
    # Dylan reckons the original DRLHP selected segments from the last RL batch.
    # We use nsteps=2048 for Atari. With 16 environments, that's an RL batch of 2048 * 16 = 32768 steps.
    # Assuming 1.5-second segments (45 steps), that's about 728 segments per batch.
    # 1000 is close enough.
    parser.add_argument('--max_segs', type=int, default=1000)
    parser.add_argument('--rollout_action_noise', type=float, default=0.5)
    parser.add_argument('--rollout_mode', choices=['primitives', 'cur_policy'], default='primitives')
    parser.add_argument('--cur_policy_randomness', choices=['sample', 'noise'], default='noise')
    parser.add_argument('--n_cur_policy', type=int, default=3)
    args = parser.parse_args()

    global_variables.segment_save_mode = args.segment_save_mode
    global_variables.max_segs = args.max_segs
    global_variables.render_segments = args.render_segments
    global_variables.rollout_action_noise = args.rollout_action_noise
    global_variables.rollout_mode = RolloutMode[args.rollout_mode]
    global_variables.rollout_randomness = RolloutRandomness[args.cur_policy_randomness]
    global_variables.n_cur_policy = args.n_cur_policy

    if args.render_every_nth_episode is None:
        if 'Fetch' in args.env:
            args.render_every_nth_episode = 17
        else:
            args.render_every_nth_episode = 20

    if args.log_dir:
        log_dir = args.log_dir
    else:
        git_rev = get_git_rev()
        run_name = args.run_name + '_' + git_rev
        log_dir = osp.join('runs', run_name)
        if osp.exists(log_dir):
            raise Exception("Log directory '%s' already exists" % log_dir)
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    save_args(args, log_dir)

    configure_logger(log_dir)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    return args, log_dir


def configure_logger(log_dir):
    tb = logger.TensorBoardOutputFormat(log_dir)
    logger.Logger.CURRENT = logger.Logger(dir=log_dir, output_formats=[tb])