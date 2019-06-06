#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('--seeds', default='0')
parser.add_argument('--test', action='store_true')
parser.add_argument('--gpus', default='')
parser.add_argument('--extra_args', default='')
parser.add_argument('--harness_extra_args', default='')
args = parser.parse_args()

seeds = list(map(int, args.seeds.split(',')))
if args.test:
    test_args = '--n_initial_prefs 0 --n_initial_demos 0 --pretrain_reward_predictor_seconds 0 --min_label_interval_seconds 1'
else:
    test_args = ''

rl_envs = [
    ('seaquest', 'SeaquestDeepMind-v0'),
    ('enduro', 'EnduroDeepMindNoSpeedo-v0'),
    ('breakout', 'BreakoutDeepMind-v0'),
    ('fetch', 'FetchPickAndPlace-Repeat1-ContGripper-WithGripObs-InfInitialBlockPos-FixedGoal-Delta-GripperBonuses-v0'),
    ('lunarlander', 'LunarLanderStatefulStats-v0'),
]

prefs_envs = [
    ('seaquest', 'SeaquestDeepMindDense-v0'),
    ('enduro', 'EnduroDeepMindNoSpeedo-v0'),
    ('breakout', 'BreakoutDeepMindDense-v0'),
    ('fetchpp', 'FetchPickAndPlace-Repeat1-ContGripper-WithGripObs-InfInitialBlockPos-FixedGoal-Delta-GripperBonuses-v0'),
    ('lunarlander', 'LunarLanderStatefulStats-v0'),
]

# RL using environment reward
for seed in seeds:
    for env_shortname, env_id in rl_envs:
        run_name = f"{env_shortname}-{seed}-rl"
        print(f"python3 scripts/train/auto_train_rl.py {seed} {env_id} {run_name} --gpus '{args.gpus}'")

for seed in seeds:
    for env_shortname, env_id in prefs_envs:
        run_name = f"{env_shortname}-{seed}"
        if args.test:
            run_name += '-test'

        if 'Fetch' in env_id:
            rollout_length_seconds = 1.0
        elif 'Breakout' in env_id:
            # Long enough to stretch from hitting the ball to the ball bouncing off a block
            rollout_length_seconds = 1.5
        else:
            rollout_length_seconds = 1.0

        extra_args = f"{args.extra_args} --rollout_length_seconds {rollout_length_seconds}"

        # DRLHP
        print("python3 scripts/train/auto_train_prefs.py "
              f"{env_id} reward_only drlhp {run_name}-drlhp-{args.name} --seed {seed} --disable_redo --extra_args ' {extra_args}' {test_args} --gpus '{args.gpus}' {args.harness_extra_args}")

        # DRLHP with label rate decay
        print("python3 scripts/train/auto_train_prefs.py "
              f"{env_id} reward_only drlhp {run_name}-drlhpd-{args.name} --seed {seed} --disable_redo --extra_args ' {extra_args}' {test_args} --gpus '{args.gpus}' --decay_label_rate {args.harness_extra_args}")

        # SDRLHP
        print("python3 scripts/train/auto_train_prefs.py "
              f"{env_id} reward_only demonstrations {run_name}-sdrlhp-{args.name} --seed {seed} --disable_redo --extra_args ' {extra_args}' {test_args} --gpus '{args.gpus}' {args.harness_extra_args}")

        # SDRLHP-NP
        np_args = ''
        if 'Breakout' in env_id:
            np_args += '--cur_policy_randomness correlated_random_action --rollout_random_action_prob 0.8 --rollout_random_correlation 0.7'
        if 'Enduro' in env_id:
            np_args += '--cur_policy_randomness correlated_random_action --rollout_random_action_prob 1.0 --rollout_random_correlation 0.99'
        print("python3 scripts/train/auto_train_prefs.py "
              f"{env_id} reward_only sdrlhpnp {run_name}-sdrlhpnp-{args.name} --seed {seed} --disable_redo --extra_args ' {extra_args} {np_args}' {test_args} --gpus '{args.gpus}' {args.harness_extra_args}")

        # SDRLHP-NP-DRLHP
        print("python3 scripts/train/auto_train_prefs.py "
              f"{env_id} reward_only sdrlhpnp-drlhp {run_name}-sdrlhpnp-drlhp-{args.name} --seed {seed} --disable_redo --extra_args ' {extra_args} {np_args}' {test_args} --gpus '{args.gpus}' {args.harness_extra_args}")

        # SDRLHP-NP with label rate decay
        print("python3 scripts/train/auto_train_prefs.py "
              f"{env_id} reward_only sdrlhpnp {run_name}-sdrlhpnpd-{args.name} --seed {seed} --disable_redo --extra_args ' {extra_args} {np_args}' {test_args} --gpus '{args.gpus}' --decay_label_rate {args.harness_extra_args}")

        if 'lunarlander' in env_shortname or 'fetch' in env_shortname:
            redo = '--disable_redo'
        else:
            redo = ''
        # Behavioral cloning
        print("python3 scripts/train/auto_train_prefs.py "
              f"{env_id} bc_only demonstrations {run_name}-bc-{args.name} --seed {seed} {redo} --extra_args ' {extra_args}' {test_args} --gpus '{args.gpus}' {args.harness_extra_args}")

        # SDRLHP + behavioral cloning
        print("python3 scripts/train/auto_train_prefs.py "
              f"{env_id} reward_plus_bc demonstrations {run_name}-sdrlhp-bc-{args.name} --seed {seed} {redo} --extra_args ' {extra_args}' {test_args} --gpus '{args.gpus}' {args.harness_extra_args}")

        # Behavioral cloning on rollouts from SDRLHP-NP
        print("python3 scripts/train/auto_train_prefs.py "
              f"{env_id} bc_only sdrlhpnp {run_name}-bcnp-{args.name} --seed {seed} --disable_redo --extra_args ' {extra_args} {np_args}' {test_args} --gpus '{args.gpus}' {args.harness_extra_args}")
