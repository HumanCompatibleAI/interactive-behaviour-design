import argparse
import faulthandler
import multiprocessing
import os
import subprocess
import sys
import time

import easy_tf_log
import gym
import numpy as np
from cloudpickle import cloudpickle
from gym.wrappers import Monitor

from baselines.common.vec_env.vec_normalize import VecNormalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from policies.td3 import TD3Policy, DemonstrationsBuffer
from subproc_vec_env_custom import CustomSubprocVecEnv
from utils import get_git_rev
from wrappers.fetch_pick_and_place_register import register
from wrappers.wrappers import LogEpisodeStats
from wrappers.wrappers_debug import DrawActions, DrawRewards, DrawObses
from wrappers.fetch_pick_and_place import RandomInitialPosition

faulthandler.enable()


def get_action_smooth(obs):
    assert obs.shape == (7,)
    gripper_to_block = obs[:3]
    block_to_target = obs[3:6]
    gripper_width = obs[6]
    if (np.linalg.norm(gripper_to_block) > 0.01 and
            np.linalg.norm(block_to_target) > 0.1):  # Don't open grippers if blocking is slipping near target
        if np.linalg.norm(gripper_to_block[:2]) > 0.05:
            z = 0
        else:
            z = gripper_to_block[2]
        action = np.concatenate([3 * gripper_to_block[:2], [3 * z, 1]])
    elif gripper_width > 0.05:
        action = np.array([0, 0, 0, -1])
    else:
        action = np.concatenate([3 * block_to_target, [-1]])
    return action


def get_action_jerky(obs):
    assert obs.shape == (7,)
    gripper_to_block = obs[:3]
    block_to_target = obs[3:6]
    gripper_width = obs[6]
    if (np.linalg.norm(gripper_to_block) > 0.03 and
            np.linalg.norm(block_to_target) > 0.1):  # Don't open grippers if blocking is slipping near target
        di = np.argmax(np.abs(gripper_to_block))
        action = np.array([0., 0., 0., 1.])
        action[di] = 0.15 * np.sign(gripper_to_block[di])
    elif gripper_width > 0.05:
        action = np.array([0, 0, 0, -1])
    else:
        di = np.argmax(np.abs(block_to_target))
        action = np.array([0., 0., 0., -1.])
        action[di] = 0.15 * np.sign(block_to_target[di])
    return action


def gen_demonstrations(env_id, log_dir, n_demonstrations, demonstrations_buffer: DemonstrationsBuffer):
    env = gym.make(env_id)
    env = RandomInitialPosition(env)
    env = LogEpisodeStats(env, log_dir, '_demo')
    # env = DrawObses(env)
    # env = DrawActions(env)
    # env = Monitor(env, video_callable=lambda n: True, directory=log_dir, uid=111)

    for n in range(n_demonstrations):
        print(f"Generating demonstration {n}...")
        obs, done = env.reset(), False
        s = 0
        action = get_action_jerky(obs)
        while not done:
            demonstrations_buffer.store(obs, action)
            obs, reward, done, info = env.step(action)
            if s % 7 == 0:
                action = get_action_jerky(obs)
            s += 1


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
    test_env = LogEpisodeStats(test_env, os.path.join(log_dir, 'env'), '_test')
    test_env = Monitor(test_env, video_callable=lambda n: n % 7 == 0, directory=log_dir, uid=999)
    policy.test_env = test_env
    policy.init_logger(os.path.join(log_dir, 'test'))

    while True:
        policy.load_checkpoint(os.path.join(log_dir, 'model.ckpt'))
        policy.test_agent(n=1)
        n_test_eps_val.value += 1
        sys.stdout.flush()
        sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir')
    parser.add_argument('alg', choices=['sac', 'td3'])
    parser.add_argument('env_id')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', default='0.2')
    parser.add_argument('--noise_sigma', default='0.2 0.2 0.2 0.2')
    parser.add_argument('--polyak', type=float, default=0.999995)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--normalize', choices=['n', 'r', 'o', 'ro'], default='n')
    parser.add_argument('--batches_per_cycle', type=int, default=40)
    parser.add_argument('--mode', choices=['bc', 'rbc'], default='rbc')
    parser.add_argument('--n_demos', type=int, default=100)
    parser.add_argument('--l2_coef', type=float, default=1e-4)
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    record_context(os.path.join(args.log_dir, 'context.txt'), args)
    alpha = list(map(float, args.alpha.split(' ')))
    noise_sigma = np.array(list(map(float, args.noise_sigma.split(' '))))

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
    train_env = CustomSubprocVecEnv(env_fns=[lambda n=n: env_fn(seed=(n_envs * args.seed + n), log_dir=args.log_dir)
                                             for n in range(n_envs)])
    obs_space = train_env.observation_space
    act_space = train_env.action_space

    if args.normalize != 'n':
        train_env = VecNormalize(train_env,
                                 ob=('o' in args.normalize),
                                 ret=('r' in args.normalize))
        train_env.reset_one_env = train_env.venv.reset_one_env

    if args.alg == 'sac':
        raise Exception()
    elif args.alg == 'td3':
        def make_policy_fn():
            return TD3Policy('dummyname',
                             args.env_id,
                             obs_space,
                             act_space,
                             n_envs=n_envs,
                             # Matches https://arxiv.org/pdf/1802.09464.pdf
                             rollouts_per_worker=2,
                             batch_size=256,
                             cycles_per_epoch=50,
                             batches_per_cycle=args.batches_per_cycle,
                             noise_sigma=noise_sigma,
                             polyak=args.polyak,
                             pi_lr=args.lr,
                             q_lr=args.lr,
                             l2_coef=args.l2_coef)
    else:
        raise RuntimeError()

    policy = make_policy_fn()
    policy.init_logger(os.path.join(args.log_dir, 'train'))
    policy.set_training_env(train_env)
    policy.save_checkpoint(os.path.join(args.log_dir, 'model.ckpt'))

    gen_demonstrations(args.env_id, os.path.join(args.log_dir, 'demos'), args.n_demos, policy.demonstrations_buffer)

    ctx = multiprocessing.get_context('spawn')
    n_test_eps_val = ctx.Value('i')
    test_eps_proc = ctx.Process(target=run_test_env,
                                args=(cloudpickle.dumps(make_policy_fn), args.env_id, args.log_dir, n_test_eps_val))
    test_eps_proc.start()

    logger = easy_tf_log.Logger()
    logger.set_log_dir(os.path.join(args.log_dir, 'train_time'))
    last_cycle_n = None
    while policy.cycle_n < 1000:
        if args.mode == 'bc':
            policy.train_bc()
        elif args.mode == 'rbc':
            policy.train()
        else:
            raise Exception()
        if policy.cycle_n != last_cycle_n:
            policy.save_checkpoint(os.path.join(args.log_dir, 'model.ckpt'))
            last_cycle_n = policy.cycle_n
            logger.measure_rate('cycles', policy.cycle_n, 'cycles_per_second')

    last_n_test_eps = n_test_eps_val.value
    while n_test_eps_val.value < last_n_test_eps + 25:
        time.sleep(1)

    train_env.close()
    test_eps_proc.terminate()


if __name__ == '__main__':
    main()
