#!/usr/bin/env python3

import faulthandler
import glob
import multiprocessing
import os
import os.path as osp
import pickle
import platform
import random
import re
import threading
import time
from multiprocessing import Queue, Process
from typing import List

import easy_tf_log
import gym
import numpy as np
import psutil
import tensorflow as tf
import wandb
from cloudpickle import cloudpickle
from gym.envs.atari import AtariEnv
from gym.envs.box2d import LunarLander
from gym.envs.mujoco import MujocoEnv
from gym.envs.robotics import FetchEnv
from gym.envs.robotics.robot_env import RobotEnv
from matplotlib.pyplot import figure, clf, plot, savefig, grid, legend, ylim

import global_variables
import throttler
from a2c.policies import mlp, nature_cnn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv as SubprocVecEnvBaselines
from basicfetch import basicfetch
from checkpointer import Checkpointer
from classifier_buffer import ClassifierDataBuffer
from classifier_collection import ClassifierCollection
from drlhp.pref_db import PrefDBTestTrain
from drlhp.reward_predictor import RewardPredictor, PredictedRewardNormalization
from drlhp.reward_predictor_core_network import net_mlp, net_cnn
from drlhp.training import drlhp_train_loop, drlhp_load_loop
from env import make_envs
from params import parse_args
from policies.fetch import FetchAction, FetchTD3Policy
from policies.policy_collection import PolicyCollection
from policies.ppo import PPOPolicy
from policies.td3_test_long import Oracle
from policy_rollouter import PolicyRollouter
from reward_switcher import RewardSelector
from rollouts import RolloutsByHash, CompressedRollout
from subproc_vec_env_custom import SubprocVecEnvNoAutoReset
from utils import find_latest_checkpoint, MemoryProfiler, configure_cpus, \
    load_cpu_config, register_debug_handler, get_available_gpu_ns, ObsRewardTuple
from web_app.app import run_web_app
from web_app.comparisons import monitor_segments_dir_loop, write_segments_loop
from wrappers import seaquest_reward, fetch_pick_and_place_register, lunar_lander_reward, breakout_reward, enduro, \
    fetch_reach, fetch_block_stacking
from wrappers.util_wrappers import ResetMode, ResetStateCache, VecLogRewards, DummyRender, \
    VecSaveSegments

run = wandb.init(project="interactive-behaviour-design")
os.environ['OMPI_MCA_btl_base_warn_component_unused'] = '0'
os.environ['OPENAI_LOG_FORMAT'] = ''
tf.logging.set_verbosity(tf.logging.ERROR)

lunar_lander_reward.register()
seaquest_reward.register()
breakout_reward.register()
enduro.register()
fetch_pick_and_place_register.register()
fetch_reach.register()
fetch_block_stacking.register()
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


def load_reference_trajectory(env_id):
    if env_id == 'FetchReach-CustomActionRepeat5ActionLimit0.2-v0':
        with open('reference_trajectories/reach_reference_trajectory.pkl', 'rb') as f:
            traj = pickle.load(f)
    else:
        traj = None
    return traj


def calculate_slopes(num_list):
    num_list = np.array(num_list)
    return num_list[1:] - num_list[:-1]

def predict_reference_trajectory_reward_loop(reference_trajectory: List[ObsRewardTuple],
                                             reward_predictor: RewardPredictor,
                                             log_dir):
    reference_trajectory_rewards_log_dir = os.path.join(log_dir, 'reference_trajectory_rewards')
    true_rewards = np.array([tup.reward for tup in reference_trajectory])
    obses = np.array([tup.obs for tup in reference_trajectory])
    logger = easy_tf_log.Logger(reference_trajectory_rewards_log_dir)
    log_file = open(os.path.join(reference_trajectory_rewards_log_dir, 'predicted_rewards.txt'), 'w')
    imgs_dir = os.path.join(log_dir, 'reference_trajectory_reward_images')
    os.makedirs(imgs_dir)

    test_n = 0

    figure()

    while True:
        predicted_rewards_unnormalized = reward_predictor.unnormalized_rewards(obses)[0]
        predicted_rewards_normalized = reward_predictor.normalized_rewards(obses, update_normalization=False)
        assert true_rewards.shape == predicted_rewards_unnormalized.shape, (predicted_rewards_unnormalized.shape, true_rewards.shape)

        predicted_rewards_rescaled = np.copy(predicted_rewards_unnormalized)
        predicted_rewards_rescaled -= np.min(predicted_rewards_rescaled)
        predicted_rewards_rescaled /= np.max(predicted_rewards_rescaled)
        predicted_rewards_rescaled *= (np.max(true_rewards) - np.min(true_rewards))
        predicted_rewards_rescaled += np.min(true_rewards)

        true_slopes = np.sign(calculate_slopes(true_rewards))
        predicted_slopes = np.sign(calculate_slopes(predicted_rewards_unnormalized))
        slope_match = np.sum(true_slopes == predicted_slopes)

        logger.logkv('reference_trajectory/predicted_reward_mean', np.mean(predicted_rewards_unnormalized))
        logger.logkv('reference_trajectory/predicted_reward_std', np.std(predicted_rewards_unnormalized))
        logger.logkv('reference_trajectory/slope_match', slope_match)

        log_file.write(f'Test {test_n}\n')
        for i in range(len(predicted_rewards_unnormalized)):
            log_file.write(f'{true_rewards[i]} {predicted_rewards_unnormalized[i]}\n')
        log_file.write('\n')

        clf()
        grid()
        plot(predicted_rewards_unnormalized, label='Predicted rewards')
        plot(predicted_rewards_normalized, label='Predicted rewards (normalized)')
        plot(predicted_rewards_rescaled, label='Predicted rewards (rescaled)')
        plot(true_rewards, label='Environment rewards')
        ylim([-4, 3])
        legend()
        savefig(os.path.join(imgs_dir, '{}.png'.format(test_n)))

        test_n += 1
        time.sleep(10)


def main():
    register_debug_handler()

    args, log_dir = parse_args()
    # check_env(args.env)

    gpu_ns = get_available_gpu_ns()
    configure_cpus(log_dir)
    load_cpu_config(log_dir, 'main')

    np.random.seed(args.seed)
    random.seed(args.seed)

    if platform.system() == 'Darwin':
        raise Exception(
            "Due to fork restrictions on macOS core libraries, macOS is not supported")

    throttler.init(log_dir, 0)

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
        policy_type = FetchTD3Policy
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
    train_env, test_env, demo_env = make_envs(env_id=args.env,
                                              num_env=args.n_envs, seed=args.seed,
                                              log_dir=log_dir,
                                              reset_state_server_queue=reset_state_cache.queue_to_training,
                                              reset_state_receiver_queue=reset_state_cache.queue_from_training,
                                              reset_mode_value=training_reset_mode_value,
                                              episode_obs_queue=obs_queue,
                                              segments_queue=segments_queue,
                                              render_segments=args.render_segments,
                                              render_every_nth_episode=args.render_every_nth_episode,
                                              save_states=(not args.no_save_states),
                                              policy_type=policy_type)

    reset_state_cache.start_saver_receiver()

    global_variables.env_creation_lock = threading.Lock()

    if args.no_render_demonstrations:
        demo_env = DummyRender(demo_env)
    demo_env.reset()

    # So that the function can be pickled without having to pickle the env itself
    obs_space = train_env.observation_space
    ac_space = train_env.action_space

    policy_args = parse_policy_args(args.policy_args)

    def make_policy(name, **kwargs):
        kwargs = dict(kwargs)
        kwargs.update(policy_args)
        return policy_type(name=name,
                           env_id=args.env,
                           obs_space=obs_space,
                           ac_space=ac_space,
                           n_envs=args.n_envs,
                           **kwargs)

    demonstration_rollouts = RolloutsByHash(maxlen=args.demonstrations_buffer_len)

    if args.generate_expert_demonstrations:
        if 'Fetch' not in args.env:
            raise Exception(f"Unsure how to generate expert demonstrations for env '{args.env}'")
        print("Generating expert demonstrations...")
        oracle = Oracle(mode='smooth')
        for _ in range(20):
            obses, actions = [], []
            obs, done = demo_env.reset(), False
            while not done:
                action = oracle.get_action(obs)
                obses.append(obs)
                actions.append(action)
                obs, _, done, _ = demo_env.step(action)
            oracle.reset()
            # noinspection PyTypeChecker
            rollout = CompressedRollout(obses=obses, actions=actions,
                                        final_env_state=None, rewards=None, frames=None)
            demonstration_rollouts[rollout.hash] = rollout

    policies = PolicyCollection(make_policy, log_dir, demonstration_rollouts, args.seed, test_env)
    if args.add_manual_fetch_policies:
        for action in FetchAction:
            policies.add_policy(str(action), policy_kwargs={'fetch_action': action})

    demonstration_rollouts_dir = osp.join(log_dir, 'demonstrations')
    os.makedirs(demonstration_rollouts_dir)
    demonstrations_reset_mode_value = multiprocessing.Value('i', ResetMode.USE_ENV_RESET.value)
    policy_rollouter = PolicyRollouter(demo_env, demonstration_rollouts_dir,
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
    classifier_data_buffer = ClassifierDataBuffer(video_dir=os.path.join(log_dir, 'train_env'),
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

    if args.load_policy_ckpt_dir:
        meta_paths = glob.glob(os.path.join(args.load_policy_ckpt_dir, 'policy*.meta'))
        policy_names = {re.search('policy-([^-]*)-', os.path.basename(p)).group(1)
                        for p in meta_paths}
        for policy_name in policy_names:
            policies.add_policy(policy_name)
            last_ckpt_name = find_latest_checkpoint(
                os.path.join(args.load_policy_ckpt_dir, 'policy-{}-'.format(policy_name)))
            policies.policies[policy_name].load_checkpoint(last_ckpt_name)

    # classifier_data_buffer is passed because it's what the classifiers train on
    classifier = ClassifierCollection(classifier_data_buffer, log_dir,
                                      classifier_network, train_env.observation_space.shape)
    for label_name in classifier_data_buffer.get_label_names():
        print("Adding classifier for label '{}'...".format(label_name))
        classifier.add_classifier(label_name)

    if global_variables.segment_save_mode == 'multi_env':
        train_env = VecSaveSegments(train_env, segments_queue)
    train_env = VecLogRewards(train_env, os.path.join(log_dir, 'vec_rewards'))

    policies.train_env = train_env

    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    pref_db_ckpt_name = 'pref_dbs.pkl'
    checkpointer = Checkpointer(ckpt_dir, policies, classifier, pref_db, pref_db_ckpt_name)
    checkpointer.checkpoint()

    reward_normalization = global_variables.predicted_reward_normalization
    def make_reward_predictor_fn(name, gpu_n):
        return RewardPredictor(network=reward_predictor_network, network_args=reward_predictor_network_args,
                               log_dir=log_dir, obs_space=obs_space,
                               r_std=reward_predictor_std,
                               name=name, gpu_n=gpu_n,
                               reward_normalization=reward_normalization)

    if gpu_ns:
        reward_predictor_inference_gpu_n = gpu_ns[0]
    else:
        reward_predictor_inference_gpu_n = None
    reward_predictor = make_reward_predictor_fn('inference', gpu_n=reward_predictor_inference_gpu_n)

    # Reward predictor training loop
    # Loads preferences, trains, saves checkpoint
    reward_predictor_ckpt_name = 'reward_predictor.ckpt'
    context = multiprocessing.get_context('spawn')
    run_drlhp_training = context.Value('B', 0)
    if gpu_ns:
        if len(gpu_ns) > 1:
            reward_predictor_training_gpu_n = gpu_ns[1]
        else:
            reward_predictor_training_gpu_n = gpu_ns[0]
    else:
        reward_predictor_training_gpu_n = None
    drlhp_train_process = context.Process(
        target=drlhp_train_loop,
        args=(cloudpickle.dumps(make_reward_predictor_fn),
              run_drlhp_training,
              os.path.join(ckpt_dir, pref_db_ckpt_name),
              os.path.join(ckpt_dir, reward_predictor_ckpt_name),
              log_dir,
              reward_predictor_training_gpu_n))
    drlhp_train_process.start()
    global_variables.pids_to_proc_names[drlhp_train_process.pid] = 'reward_predictor_training'

    # Loads checkpoint generated by train loop every so often
    drlhp_sync_thread = threading.Thread(target=drlhp_load_loop,
                                         args=(reward_predictor,
                                               os.path.join(ckpt_dir, reward_predictor_ckpt_name),
                                               log_dir))
    drlhp_sync_thread.start()

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
        last_ckpt_name = find_latest_checkpoint(os.path.join(args.load_classifier_ckpt, 'classifiers-'))
        classifier.load_checkpoint(last_ckpt_name)

    reward_selector = RewardSelector(classifier, reward_predictor)
    global_variables.reward_selector = reward_selector

    if isinstance(train_env.unwrapped, SubprocVecEnvNoAutoReset):
        for n in range(train_env.num_envs):
            train_env.reset_one_env(n)
    elif isinstance(train_env.unwrapped, SubprocVecEnvBaselines):
        train_env.reset()
    else:
        raise RuntimeError("train_env is neither SubprocVecEnvNoAutoReset nor SubprocVecEnvBaselines")

    os.makedirs(os.path.join(wandb.run.dir, 'media'))
    os.symlink(os.path.join(log_dir, 'test_env'), os.path.join(wandb.run.dir, 'media', 'test_env'))
    os.symlink(os.path.join(log_dir, 'train_env'), os.path.join(wandb.run.dir, 'media', 'train_env'))

    time.sleep(5)  # Give time for processes to start
    mp = MemoryProfiler(pid=os.getpid(), log_path=os.path.join(log_dir, f'memory-self.txt'))
    mp.start()
    mp = MemoryProfiler(pid=os.getpid(), include_children=True,
                        log_path=os.path.join(log_dir, f'memory-self-with-children.txt'))
    mp.start()
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        proc_name = global_variables.pids_to_proc_names.get(child.pid)
        if proc_name is None:
            proc_name = child.pid
        log_name = f'memory-{proc_name}.txt'
        mp = MemoryProfiler(pid=child.pid, log_path=os.path.join(log_dir, log_name))
        mp.start()

    reference_trajectory = load_reference_trajectory(args.env)
    if reference_trajectory is not None:
        predict_reference_trajectory_reward_thread = threading.Thread(
            target=lambda: predict_reference_trajectory_reward_loop(reference_trajectory, reward_predictor, log_dir))
        predict_reference_trajectory_reward_thread.start()

    # Run

    run_web_app(classifiers=classifier,
                policies=policies,
                reward_selector=reward_selector,
                experience_buffer=classifier_data_buffer,
                log_dir=log_dir,
                port=args.port,
                pref_db=pref_db,
                demo_env=demo_env,
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

    train_env.close()


def parse_policy_args(policy_args_str):
    """
    foo=bar,baz=qux -> {foo: bar, baz: qux}
    """
    if not policy_args_str:
        return {}

    policy_args = {k: v
                   for k, v in [a.split('=')
                                for a in policy_args_str.split(',')]}
    for k, v in policy_args.items():
        if v == 'True':
            policy_args[k] = True
        elif v == 'False':
            policy_args[k] = False
        elif any([c in v for c in ['.', 'e']]):
            policy_args[k] = float(v)
        else:
            policy_args[k] = int(v)

    return policy_args


if __name__ == '__main__':
    main()
