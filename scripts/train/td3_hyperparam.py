env_id = 'FetchPickAndPlace-Repeat1-ContGripper-WithGripObs-InfInitialBlockPos-FixedGoal-Delta-GripperBonuses-v0'
for polyak in [0.95, 0.999995]:
    for batches_per_cycle in [50, 200]:
        for seed in [0, 1, 2, 3, 4]:
                run_name = f'fetch-hyperparam-polyak{polyak}-bpc{batches_per_cycle}-seed{seed}'
                run_name = run_name.replace('.', 'p')  # tmux name
                print(f"python3 scripts/train/auto_train_rl.py {seed} {env_id} {run_name} "
                      f"--policy_args polyak={polyak};batches_per_cycle={batches_per_cycle}")
