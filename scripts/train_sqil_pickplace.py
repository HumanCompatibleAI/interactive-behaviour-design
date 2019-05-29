import argparse
import faulthandler
import multiprocessing
import os
import subprocess
import sys

import gym
import numpy as np
import tensorflow as tf
from cloudpickle import cloudpickle
from gym.wrappers import Monitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from policies.td3 import TD3Policy
from policies.base_policy import PolicyTrainMode
from policies.td3_test import gen_demonstrations, Oracle
from subproc_vec_env_custom import SubprocVecEnvNoAutoReset
from utils import get_git_rev
from wrappers.fetch_pick_and_place_register import register
from wrappers.util_wrappers import SaveEpisodeStats
from wrappers.wrappers_debug import DrawActions, DrawRewards

faulthandler.enable()


def record_context(path, args):
    git_info = get_git_rev()
    if os.path.exists('.git'):
        git_info += '\n\n'
        git_info += subprocess.check_output(['git', 'status']).decode() + '\n\n'
        git_info += subprocess.check_output(['git', 'diff']).decode() + '\n\n'
    with open(path, 'w') as f:
        f.write(git_info)
        f.write(' '.join(sys.argv))
        f.write('\n\n')
        f.write(str(args))
        f.write('\n\n')


def run_test_env(policy_fn_pickle, env_id, log_dir, n_test_eps_val: multiprocessing.Value):
    policy = cloudpickle.loads(policy_fn_pickle)()
    register()
    test_env = gym.make(env_id)
    test_env = DrawActions(test_env)
    test_env = DrawRewards(test_env)
    test_env = SaveEpisodeStats(test_env, os.path.join(log_dir, 'env'), '_test')
    test_env = Monitor(test_env, video_callable=lambda n: n % 7 == 0, directory=log_dir, uid=999)
    policy.test_env = test_env
    policy.init_logger(os.path.join(log_dir, 'test'))

    while True:
        ckpt = tf.train.latest_checkpoint(log_dir)
        policy.load_checkpoint(ckpt)
        policy.test_agent(n=1)
        n_test_eps_val.value += 1
        sys.stdout.flush()
        sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir')
    parser.add_argument('--env_id', default='FetchPickAndPlace-Repeat1-ContGripper-WithGripObs-1InitialBlockPos-FixedGoal-Delta-GripperBonuses-v0')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--polyak', type=float, default=0.9995)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--goal_state_reward', type=int)
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    record_context(os.path.join(args.log_dir, 'context.txt'), args)

    register()
    first_env_semaphore = multiprocessing.Semaphore()

    def env_fn(seed, log_dir):
        env = gym.make(args.env_id)
        env.unwrapped.seed(seed)
        if first_env_semaphore.acquire(timeout=0):
            env = DrawActions(env)
            env = DrawRewards(env)
            # 7 is coprime with 5 (the number of initial positions),
            # ensuring we get to see videos starting from all initial positions
            env = Monitor(env, video_callable=lambda n: n % 7 == 0, directory=log_dir)
        return env

    n_envs = 19
    train_env = SubprocVecEnvNoAutoReset(env_fns=[lambda env_n=env_n: env_fn(seed=((args.seed * n_envs) + env_n),
                                                                             log_dir=args.log_dir)
                                                  for env_n in range(n_envs)])

    obs_space = train_env.observation_space
    act_space = train_env.action_space

    def make_policy_fn():
        return TD3Policy('dummyname',
                         args.env_id,
                         obs_space,
                         act_space,
                         n_envs=n_envs,
                         hidden_sizes=(256, 256, 256, 256),
                         train_mode=PolicyTrainMode.SQIL_ONLY,
                         pi_lr=args.lr,
                         q_lr=args.lr,
                         polyak=args.polyak,
                         seed=args.seed
                         )

    policy = make_policy_fn()
    policy.init_logger(os.path.join(args.log_dir, 'train'))
    policy.set_training_env(train_env)
    policy.save_checkpoint(os.path.join(args.log_dir, 'model.ckpt'))

    ctx = multiprocessing.get_context('spawn')
    n_test_eps_val = ctx.Value('i')
    test_eps_proc = ctx.Process(target=run_test_env,
                                args=(cloudpickle.dumps(make_policy_fn), args.env_id, args.log_dir, n_test_eps_val))
    test_eps_proc.start()

    oracle = Oracle('smooth')
    n_demos = 1
    gen_demonstrations(args.env_id, os.path.join(args.log_dir, 'demos'), n_demos, policy.demonstrations_buffer, oracle)

    if args.goal_state_reward:
        b = policy.demonstrations_buffer
        i = b.ptr - 1
        obs1, action, reward, obs2, done = b.obs1_buf[i], b.acts_buf[i], b.rews_buf[i], b.obs2_buf[i], b.done_buf[i]
        assert done
        goal_state = obs2
        stay_put_action = np.zeros(act_space.shape)
        policy.demonstrations_buffer.store(obs=goal_state,
                                           act=stay_put_action,
                                           rew=args.goal_state_reward,
                                           # should be ignored because done = True; set this to something really large
                                           # so that if we're wrong we see an obvious failure
                                           next_obs=sys.maxsize * np.ones_like(goal_state),
                                           done=True)
        policy.monitor_q_s = [goal_state]
        policy.monitor_q_a = [stay_put_action]

    last_cycle_n = None
    while policy.cycle_n < 1000:
        policy.train()
        if policy.cycle_n != last_cycle_n:
            print("Cycle", policy.cycle_n)
            policy.save_checkpoint(os.path.join(args.log_dir, 'model.ckpt'))
            last_cycle_n = policy.cycle_n

    train_env.close()
    test_eps_proc.terminate()


if __name__ == '__main__':
    main()
