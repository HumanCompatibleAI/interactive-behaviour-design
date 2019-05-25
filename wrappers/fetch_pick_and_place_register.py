"""
Imports are done strangely here so that we can quickly query all the environment names for train_all.py without having to import anything
"""

from wrappers.fetch_pick_and_place import FixedGoal, FixedBlockInitialPositions


def make_env(include_grip_obs, repeat_n, early_termination, binary_gripper, grip_close_bonus, grip_open_bonus,
             reward_mode, n_initial_block_positions, fixed_goal):
    from gym.envs.robotics import FetchPickAndPlaceEnv
    from gym.wrappers import FlattenDictWrapper

    from wrappers.fetch_pick_and_place import FetchStatsWrapper, FetchPickAndPlaceRewardWrapper, \
        FetchPickAndPlaceObsWrapper
    from wrappers.util_wrappers import RepeatActions

    env = FetchPickAndPlaceEnv()
    env = FixedGoal(env, fixed_goal=fixed_goal)
    env = FixedBlockInitialPositions(env, n_initial_block_positions=n_initial_block_positions)
    env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env = FetchStatsWrapper(env)
    env = FetchPickAndPlaceRewardWrapper(env,
                                         early_termination=early_termination,
                                         binary_gripper=binary_gripper,
                                         grip_close_bonus=grip_close_bonus,
                                         grip_open_bonus=grip_open_bonus,
                                         reward_mode=reward_mode)
    env = FetchPickAndPlaceObsWrapper(env, include_grip_obs)
    env = RepeatActions(env, repeat_n)
    return env


def enumerate_envs(register=False):
    repeat_ns = [('Repeat1', 1), ('Repeat3', 3)]
    binary_grippers = [('BinaryGripper', True), ('ContGripper', False)]
    include_grip_obss = [('WithGripObs', True), ('NoGripObs', False)]
    n_initial_block_positionss = [('1InitialBlockPos', 1),
                                  ('5InitialBlockPos', 5),
                                  ('10InitialBlockPos', 10),
                                  ('InfInitialBlockPos', None)]
    fixed_goals = [('FixedGoal', True), ('RandomGoal', False)]
    modes = [('Delta', 'delta2'), ('NonDelta', 'nondelta')]
    gripper_bonuses = [('GripperBonuses', True), ('NoGripperBonus', False)]

    env_ids = []
    for s1, repeat in repeat_ns:
        for s2, binary_gripper in binary_grippers:
            for s3, include_grip_obs in include_grip_obss:
                for s4, n_initial_block_positions in n_initial_block_positionss:
                    for s5, fixed_goal in fixed_goals:
                        for s6, mode in modes:
                            for s7, gripper_bonus in gripper_bonuses:
                                env_id = f'FetchPickAndPlace-{s1}-{s2}-{s3}-{s4}-{s5}-{s6}-{s7}-v0'
                                env_ids.append(env_id)
                                if register:
                                    from gym.envs import register as gym_register
                                    entry_fn = lambda locals=dict(locals()): \
                                        make_env(reward_mode=locals['mode'], early_termination=False,
                                                 grip_open_bonus=locals['gripper_bonus'], grip_close_bonus=locals['gripper_bonus'],
                                                 binary_gripper=locals['binary_gripper'],
                                                 repeat_n=locals['repeat'],
                                                 include_grip_obs=locals['include_grip_obs'],
                                                 n_initial_block_positions=locals['n_initial_block_positions'],
                                                 fixed_goal=locals['fixed_goal'])
                                    gym_register(env_id,
                                                 entry_point=entry_fn,
                                                 max_episode_steps=250)
    return env_ids


def register():
    enumerate_envs(register=True)
