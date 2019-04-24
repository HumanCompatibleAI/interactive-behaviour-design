#!/usr/bin/env python3

import os
import random

if  'CUDA_VISIBLE_DEVICES' not in os.environ:
    if random.random() < 0.5:
        d = '0'
    else:
        d = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = d

os.environ['OMPI_MCA_btl_base_warn_component_unused'] = '0'
os.environ['OPENAI_LOG_FORMAT'] = ''

import random

import numpy as np
import psutil
from gym.envs.robotics.robot_env import RobotEnv

from basicfetch import basicfetch
from policies.fetch import FetchAction, FetchPPOPolicy
from policies.policy_collection import PolicyCollection
from policies.ppo import PPOPolicy

import faulthandler
import glob
import multiprocessing
import os.path as osp
import platform
import re
import threading
import time
import traceback
from multiprocessing import Queue, Process

import gym
from gym.envs.atari import AtariEnv
from gym.envs.box2d import LunarLander
from gym.envs.mujoco import MujocoEnv
from gym.envs.robotics import FetchEnv

import global_variables
from a2c.policies import mlp, nature_cnn
from classifier_buffer import ClassifierDataBuffer
from classifier_collection import ClassifierCollection
from drlhp.pref_db import PrefDBTestTrain
from drlhp.reward_predictor_core_network import net_mlp, net_cnn
from env import make_env
from params import parse_args
from rollouts import RolloutsByHash
from utils import find_latest_checkpoint, MemoryProfiler
from segments import monitor_segments_dir_loop, write_segments_loop
from wrappers import seaquest_reward, fetch_pick_and_place_register, lunar_lander_reward
from wrappers.util_wrappers import VecRewardSwitcherWrapper, ResetMode, ResetStateCache, VecLogRewards, DummyRender, \
    VecSaveSegments
from policy_rollouter import PolicyRollouter
from checkpointer import Checkpointer
from web_app.app import run_web_app
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

lunar_lander_reward.register()
seaquest_reward.register()
fetch_pick_and_place_register.register()
basicfetch.register()
faulthandler.enable()


def check_env(env_id):
    if any(x in env_id for x in ['LunarLander', 'Fetch']):
        if not 'DISPLAY' in os.environ:
            raise Exception(f"DISPLAY must be set for environment {env_id}")
    supported_envs = ['SeaquestDeepMindDense-v0',
                      'SeaquestDeepMind-v0',
                      'LunarLanderStatefulEarlyTermination-v0',
                      'FetchPickAndPlaceDense1-v0',
                      'FetchPickAndPlaceDense2-v0',
                      'FetchReachDense-v0',
                      'FetchBasic-v0']
    if env_id not in supported_envs:
        raise Exception(f"Env {env_id} not supported; try", ','.join(supported_envs))


def main():
    args, log_dir = parse_args()
    # check_env(args.env)

    np.random.seed(args.seed)
    random.seed(args.seed)

    if platform.system() == 'Darwin':
        raise Exception(
            "Due to fork restrictions on macOS core libraries, macOS is not supported")

    # Create env and wrappers
    segments_dir = osp.join(log_dir, 'segments')
    experience_dir = osp.join(log_dir, 'experience')
    [os.makedirs(d) for d in [segments_dir, experience_dir]]
    reset_state_cache = ResetStateCache(experience_dir)
    training_reset_mode_value = multiprocessing.Value('i', ResetMode.USE_ENV_RESET.value)
    save_state_from_proportion_through_episode_value = multiprocessing.Value('d', 0.5)
    max_episode_steps_value = multiprocessing.Value('i', 100000)
    segments_queue = Queue(maxsize=1)
    obs_queue = Queue()
    env = make_env(env_id=args.env,
                   num_env=args.n_envs, seed=args.seed,
                   experience_dir=experience_dir,
                   reset_state_server_queue=reset_state_cache.queue_to_training,
                   reset_state_receiver_queue=reset_state_cache.queue_from_training,
                   reset_mode_value=training_reset_mode_value,
                   max_episode_steps_value=max_episode_steps_value,
                   episode_proportion_value=save_state_from_proportion_through_episode_value,
                   episode_obs_queue=obs_queue,
                   segments_queue=segments_queue,
                   segments_dir=segments_dir,
                   render_segments=args.render_segments,
                   render_every_nth_episode=args.render_every_nth_episode)
    reset_state_cache.start_saver_receiver()

    global_variables.env_creation_lock = threading.Lock()

    demonstrations_env = env.demonstrations_env
    if args.no_render_demonstrations:
        demonstrations_env = DummyRender(demonstrations_env)
    demonstrations_env.reset()

    dummy_env = gym.make(args.env)
    if isinstance(dummy_env.unwrapped, (MujocoEnv, LunarLander)):
        classifier_network = mlp
        reward_predictor_network = net_mlp
        reward_predictor_network_args = {}
        reward_predictor_std = 0.05
        policy_type = PPOPolicy
    elif isinstance(dummy_env.unwrapped, (FetchEnv, RobotEnv)):  # RobotEnv for FetchBasic
        classifier_network = mlp
        reward_predictor_network = net_mlp
        reward_predictor_network_args = {}
        reward_predictor_std = 1.0
        policy_type = FetchPPOPolicy
    elif isinstance(dummy_env.unwrapped, AtariEnv):
        classifier_network = nature_cnn
        reward_predictor_network = net_cnn
        reward_predictor_network_args = {'batchnorm': False, 'dropout': 0.5}
        reward_predictor_std = 0.05
        policy_type = PPOPolicy
    else:
        raise Exception("Unknown environment type: {}".format(dummy_env))

    if args.rstd is not None:
        reward_predictor_std = args.rstd
        print("Overriding reward predictor std:", reward_predictor_std)

    # So that the function can be pickled without having to pickle the env itself
    obs_space = env.observation_space
    ac_space = env.action_space

    def make_policy(name, **kwargs):
        return policy_type(name=name,
                           env_id=args.env,
                           obs_space=obs_space,
                           ac_space=ac_space,
                           n_envs=args.n_envs,
                           **kwargs)

    demonstration_rollouts = RolloutsByHash(maxlen=args.demonstrations_buffer_len)

    policies = PolicyCollection(make_policy, log_dir, demonstration_rollouts, args.seed)
    if args.add_manual_fetch_policies:
        for action in FetchAction:
            policies.add_policy(str(action), policy_kwargs={'fetch_action': action})

    demonstration_rollouts_dir = osp.join(log_dir, 'demonstrations')
    os.makedirs(demonstration_rollouts_dir)
    demonstrations_reset_mode_value = multiprocessing.Value('i', ResetMode.USE_ENV_RESET.value)
    policy_rollouter = PolicyRollouter(demonstrations_env, demonstration_rollouts_dir,
                                       reset_state_queue_in=reset_state_cache.queue_to_demonstrations,
                                       reset_mode_value=demonstrations_reset_mode_value,
                                       log_dir=log_dir, make_policy_fn=make_policy,
                                       redo_policy=args.redo_policy, noisy_policies=args.noisy_policies,
                                       rollout_len_seconds=args.rollout_length_seconds,
                                       show_from_end_seconds=args.show_from_end)

    Process(target=monitor_segments_dir_loop, args=(segments_dir, global_variables.max_segs)).start()
    Process(target=write_segments_loop, args=[segments_queue, segments_dir]).start()

    # Create initial stuff

    if args.no_save_frames:
        frames_save_dir = None
    else:
        frames_save_dir = experience_dir
    classifier_data_buffer = ClassifierDataBuffer(video_dir=experience_dir,
                                                  save_dir=frames_save_dir)
    classifier_data_buffer.start_saving_obs_from_queue(obs_queue)

    pref_db = PrefDBTestTrain()
    if args.load_drlhp_prefs:
        pref_db.load(args.load_drlhp_prefs)

    hasLoadedDemos = False
    hasLoadedPrefs = False
    if args.load_sdrlhp_demos:
        rollouts_pkl_path = args.load_sdrlhp_demos
        if os.path.exists(rollouts_pkl_path):
            print("Loading demonstration rollouts...")
            demonstration_rollouts.load(rollouts_pkl_path)
            hasLoadedDemos = True
        else:
            print(f"Warning: {rollouts_pkl_path} not found")
    if args.load_sdrlhp_prefs:
        pref_pkl_path = args.load_sdrlhp_prefs
        if os.path.exists(pref_pkl_path):
            print("Loading preferences...")
            pref_db.load(pref_pkl_path)
            hasLoadedPrefs = True
        else:
            print(f"Warning: {pref_pkl_path} not found")
    if args.load_experience_dir:
        print("Loading classifier data...")
        try:
            classifier_data_buffer.load_from_dir(args.load_experience_dir)
        except Exception as e:
            print(e)

        pref_pkl_path = os.path.join(args.load_experience_dir, 'pref_db.pkl')
        if os.path.exists(pref_pkl_path) and not hasLoadedDemos:
            print(f"Loading preferences from {args.load_experience_dir}...")
            pref_db.load(pref_pkl_path)

        rollouts_pkl_path = os.path.join(args.load_experience_dir, 'demonstration_rollouts.pkl')
        if os.path.exists(rollouts_pkl_path) and not hasLoadedPrefs:
            print(f"Loading demonstration rollouts from {args.load_experience_dir}...")
            demonstration_rollouts.load(rollouts_pkl_path)

        print("Loading reset states...")
        reset_state_cache.load_dir(args.load_experience_dir)

        num_eps_in_experience_dir = len([ep for ep_n, ep in classifier_data_buffer.episodes.items()
                                         if os.path.exists(ep.vid_path)])
        classifier_data_buffer.num_episodes_from_exp_dir = num_eps_in_experience_dir

    run_drlhp_training = multiprocessing.Value('B', 0)

    def f():
        while True:
            if run_drlhp_training.value == 0:
                time.sleep(1.0)
                continue

            if not pref_db.train:
                print("No preferences yet; skipping DRLHP training")
                time.sleep(1.0)
                continue

            try:
                reward_switcher_wrapper.reward_predictor.train(pref_db.train.copy(),
                                                               pref_db.test.copy(),
                                                               val_interval=1)
            except:
                print("Exception while training reward predictor:")
                traceback.print_exc()
                time.sleep(1.0)

    train_thread = threading.Thread(target=f)
    train_thread.start()

    if args.load_policy_ckpt_dir:
        meta_paths = glob.glob(os.path.join(args.load_policy_ckpt_dir, 'policy*.meta'))
        policy_names = {re.search('policy-([^-]*)-', os.path.basename(p)).group(1)
                        for p in meta_paths}
        for policy_name in policy_names:
            policies.add_policy(policy_name)
            last_ckpt_name = find_latest_checkpoint(args.load_policy_ckpt_dir,
                                                    'policy-{}-'.format(policy_name))
            policies.policies[policy_name].load_checkpoint(last_ckpt_name)

    # classifier_data_buffer is passed because it's what the classifiers train on
    classifier = ClassifierCollection(classifier_data_buffer, log_dir,
                                      classifier_network, env.observation_space.shape)
    for label_name in classifier_data_buffer.get_label_names():
        print("Adding classifier for label '{}'...".format(label_name))
        classifier.add_classifier(label_name)


    if global_variables.segment_save_mode == 'multi_env':
        env = VecSaveSegments(env, segments_queue)
    env = VecLogRewards(env, os.path.join(log_dir, 'vec_rewards'))
    env = VecRewardSwitcherWrapper(env, classifier,
                                   reward_predictor_network,
                                   reward_predictor_network_args,
                                   reward_predictor_std,
                                   log_dir)
    # We need to save a reference to this so that the web interface can call methods on it
    reward_switcher_wrapper = env
    env = VecLogRewards(env, os.path.join(log_dir, 'vec_rewards_post_switcher'), postfix='_post_switcher')

    policies.env = env

    checkpointer = Checkpointer(log_dir,
                                policies, reward_switcher_wrapper.reward_predictor, classifier)
    checkpointer.checkpoint()

    if args.load_drlhp_ckpt_dir:
        last_ckpt_name = find_latest_checkpoint(args.load_drlhp_ckpt_dir, 'drlhp_reward_predictor')
        reward_switcher_wrapper.reward_predictor.load(last_ckpt_name)

    if args.load_classifier_ckpt:
        classifier_names_path = os.path.join(args.load_classifier_ckpt, 'classifier_names.txt')
        with open(classifier_names_path) as f:
            classifier_names = f.readlines()
        # strip whitespace of
        classifier_names = [c.strip() for c in classifier_names]
        print(classifier_names)
        for classifier_name in classifier_names:
            if not classifier_name in classifier.classifiers:
                classifier.add_classifier(classifier_name)
                print("Added classifier '{}'".format(classifier_name))
        last_ckpt_name = find_latest_checkpoint(args.load_classifier_ckpt, 'classifiers-')
        classifier.load_checkpoint(last_ckpt_name)

    env.reset()

    time.sleep(5)  # Give time for processes to start
    mp = MemoryProfiler(pid=-1, log_path=os.path.join(log_dir, f'memory-self.txt'))
    mp.start()
    mp = MemoryProfiler(pid=-1, include_children=True,
                        log_path=os.path.join(log_dir, f'memory-self-with-children.txt'))
    mp.start()
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        mp = MemoryProfiler(pid=child.pid, log_path=os.path.join(log_dir, f'memory-{child.pid}.txt'))
        mp.start()

    # Run

    run_web_app(classifiers=classifier,
                policies=policies,
                reward_switcher_wrapper=reward_switcher_wrapper,
                experience_buffer=classifier_data_buffer,
                log_dir=log_dir,
                port=args.port,
                pref_db=pref_db,
                demo_env=demonstrations_env,
                policy_rollouter=policy_rollouter,
                demonstration_rollouts=demonstration_rollouts,
                reset_mode_value=training_reset_mode_value,
                reset_state_cache=reset_state_cache,
                max_episode_steps_value=max_episode_steps_value,
                save_state_from_proportion_through_episode_value=save_state_from_proportion_through_episode_value,
                demonstrations_reset_mode_value=demonstrations_reset_mode_value,
                run_drlhp_training=run_drlhp_training,
                rollout_vids_dir=demonstration_rollouts_dir,
                segments_dir=segments_dir,
                checkpointer=checkpointer,
                max_demonstration_length=args.max_demonstration_length)

    env.close()


if __name__ == '__main__':
    main()
