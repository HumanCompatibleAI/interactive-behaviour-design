#!/usr/bin/env python

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import find_least_busy_gpu, get_git_rev, NoGPUsError

rl_envs = [
    ('seaquest', 'SeaquestDeepMind-v0'),
    ('enduro', 'EnduroDeepMindNoSpeedo-v0'),
    ('breakout', 'BreakoutDeepMind-v0'),
    ('fetchpp',
     'FetchPickAndPlace-Repeat1-ContGripper-WithGripObs-InfInitialBlockPos-FixedGoal-Delta-GripperBonuses-v0'),
    ('fetchr', 'FetchReach-Custom-v0'),
    ('fetchs', 'FetchBlockStackingDense-v0'),
    ('lunarlander', 'LunarLanderStatefulStats-v0'),
]

prefs_envs = [
    ('seaquest', 'SeaquestDeepMindDense-v0'),
    ('enduro', 'EnduroDeepMindNoSpeedo-v0'),
    ('breakout', 'BreakoutDeepMindDense-v0'),
    # Important: needs to be NonDelta because the reward predictor assumes the reward is purely a function of
    # the current state
    ('fetchpp',
     'FetchPickAndPlace-Repeat1-ContGripper-WithGripObs-InfInitialBlockPos-FixedGoal-NonDelta-GripperBonuses-v0'),
    ('fetchr', 'FetchReach-Custom-v0'),
    ('fetchs', 'FetchBlockStackingDense-v0'),
    ('lunarlander', 'LunarLanderStatefulStats-v0'),
]


class Experiment:
    def __init__(self, name, train_mode, segment_generation, disable_redo, decay_label_rate, no_primitives_config=''):
        self.name = name
        self.train_mode = train_mode
        self.segment_generation = segment_generation
        self.disable_redo = disable_redo
        self.decay_label_rate = decay_label_rate
        self.no_primitives_config = no_primitives_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='0')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpus')
    parser.add_argument('--extra_args', default='')
    parser.add_argument('--harness_extra_args', default='')
    args = parser.parse_args()

    git_rev = get_git_rev()

    if args.gpus is None:
        try:
            args.gpus = str(find_least_busy_gpu())
        except NoGPUsError:
            args.gpus = ''

    seeds = list(map(int, args.seeds.split(',')))
    if args.test:
        test_args = '--n_initial_prefs 0 --n_initial_demos 0 --pretrain_reward_predictor_epochs 0 ' \
                    '--min_label_interval_seconds 0.01'
    else:
        test_args = ''

    gpu_arg = f"--gpus '{args.gpus}' " if args.gpus is not None else " "
    # RL using environment reward
    for env_shortname, env_id in rl_envs:
        for seed in seeds:
            run_name = f"{env_shortname}-{seed}-rl"
            print(f"python3 scripts/train/auto_train_rl.py {seed} {env_id} {run_name} "
                  f"{gpu_arg} "
                  f"--tags rl,{env_shortname}")

    if args.extra_args:
        run_suffix = '_' + args.extra_args.lstrip().replace(' ', '_').replace('-', '').replace('.', 'p').replace('=', '_').replace(';', '_')
    else:
        run_suffix = ''
    for env_shortname, env_id in prefs_envs:
        rollout_len_config = get_rollout_len_config(env_id, args.extra_args)
        pretrain_config = get_pretrain_config(env_id, args.harness_extra_args)
        initial_prefs_config = get_initial_prefs_config(env_id, args.harness_extra_args)
        experiments = get_experiments(env_id)
        for ex in experiments:
            redo_config = f"{'--disable_redo' if ex.disable_redo else ''} "
            decay_config = f"{'--decay_label_rate' if ex.decay_label_rate else ''} "
            extra_args = f"{rollout_len_config} {ex.no_primitives_config} {args.extra_args}"
            wandb_group = f"{env_shortname}-{ex.name}{run_suffix}_{git_rev}"
            for seed in seeds:
                run_name = f"{env_shortname}-{seed}-{ex.name}{run_suffix}"
                if args.test:
                    run_name += '_test'
                cmd = ("python3 scripts/train/auto_train_prefs.py "
                       f"{env_id} {ex.train_mode} {ex.segment_generation} {run_name} "
                       f"{test_args} {redo_config} {decay_config} {pretrain_config} {initial_prefs_config} "
                       f"{args.harness_extra_args} "
                       f"--seed {seed} "
                       f"--extra_args ' {extra_args}' --gpus '{args.gpus}' "
                       f"--tags {env_shortname},{ex.name} --group {wandb_group}")
                print(cmd)


def get_initial_prefs_config(env_id, extra_args):
    if 'n_initial_prefs' in extra_args:
        return ''
    if 'Fetch' in env_id:
        n = 500
    else:
        n = 500
    return f'--n_initial_prefs {n}'


def get_pretrain_config(env_id, extra_args):
    if 'pretrain_reward_predictor_epochs' in extra_args:
        return ''
    if 'Fetch' in env_id:
        n_epochs = 200
    else:
        n_epochs = 200
    return f'--pretrain_reward_predictor_epochs {n_epochs}'


def get_rollout_len_config(env_id, extra_args):
    if 'rollout_length_seconds' in extra_args:
        return ''
    if 'Fetch' in env_id:
        rollout_length_seconds = 0.5
    elif 'Breakout' in env_id:
        # Long enough to stretch from hitting the ball to the ball bouncing off a block
        rollout_length_seconds = 1.5
    else:
        rollout_length_seconds = 1.0
    rollout_len_config = f"--rollout_length_seconds {rollout_length_seconds}"
    return rollout_len_config


def get_experiments(env_id):
    if 'Breakout' in env_id:
        no_primitives_config = '--cur_policy_randomness correlated_random_action ' \
                               '--rollout_random_action_prob 0.8 --rollout_random_correlation 0.7'
    elif 'Enduro' in env_id:
        no_primitives_config = '--cur_policy_randomness correlated_random_action ' \
                               '--rollout_random_action_prob 1.0 --rollout_random_correlation 0.99'
    else:
        no_primitives_config = ''

    if 'LunarLander' in env_id or 'Fetch' in env_id:
        imitation_learning_disable_redo = True
    else:
        imitation_learning_disable_redo = False

    experiments = list()

    experiments.append(Experiment(
        name='drlhp',
        train_mode='reward_only',
        segment_generation='drlhp',
        disable_redo=True,
        decay_label_rate=False,
    ))

    experiments.append(Experiment(
        name='drlhpd',
        train_mode='reward_only',
        segment_generation='drlhp',
        disable_redo=True,
        decay_label_rate=True
    ))

    experiments.append(Experiment(
        name='sdrlhp',
        train_mode='reward_only',
        segment_generation='demonstrations',
        disable_redo=True,
        decay_label_rate=False
    ))

    experiments.append(Experiment(
        name='sdrlhpnp',
        train_mode='reward_only',
        segment_generation='sdrlhpnp',
        disable_redo=True,
        decay_label_rate=False,
        no_primitives_config=no_primitives_config
    ))

    experiments.append(Experiment(
        name='sdrlhpnp-drlhp',
        train_mode='reward_only',
        segment_generation='sdrlhpnp-drlhp',
        disable_redo=True,
        decay_label_rate=False,
        no_primitives_config=no_primitives_config
    ))

    experiments.append(Experiment(
        name='sdrlhpnpd',
        train_mode='reward_only',
        segment_generation='sdrlhpnp',
        disable_redo=True,
        decay_label_rate=True,
        no_primitives_config=no_primitives_config
    ))

    experiments.append(Experiment(
        name='bc',
        train_mode='bc_only',
        segment_generation='demonstrations',
        disable_redo=imitation_learning_disable_redo,
        decay_label_rate=False
    ))

    experiments.append(Experiment(
        name='sdrlhp-bc',
        train_mode='reward_plus_bc',
        segment_generation='demonstrations',
        disable_redo=imitation_learning_disable_redo,
        decay_label_rate=False
    ))

    experiments.append(Experiment(
        name='bcnp',
        train_mode='bc_only',
        segment_generation='sdrlhpnp',
        disable_redo=imitation_learning_disable_redo,
        decay_label_rate=False,
        no_primitives_config=no_primitives_config
    ))

    experiments.append(Experiment(
        name='sqilnp',
        train_mode='sqil_only',
        segment_generation='sdrlhpnp',
        disable_redo=imitation_learning_disable_redo,
        decay_label_rate=False,
        no_primitives_config=no_primitives_config
    ))

    experiments.append(Experiment(
        name='sdrlhpnp-sqil',
        train_mode='reward_plus_sqil',
        segment_generation='sdrlhpnp',
        disable_redo=imitation_learning_disable_redo,
        decay_label_rate=False,
        no_primitives_config=no_primitives_config
    ))

    return experiments


if __name__ == '__main__':
    main()
