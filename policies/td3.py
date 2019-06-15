import datetime
import glob
import os
import sys
import threading
import time
from collections import defaultdict
from threading import Thread

import numpy as np
import tensorflow as tf
from spinup.algos.td3 import core
from spinup.algos.td3.core import get_vars

import global_variables
from baselines.common.running_stat import RunningStat
from baselines.common.vec_env import VecEnv
from baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
from utils import TimerContext, LimitedRunningStat

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from policies.base_policy import Policy, PolicyTrainMode, EpisodeRewardLogger
from rollouts import RolloutsByHash

SQIL_REWARD = 5


class Batch:
    def __init__(self, obs1, obs2, acts, rews, done):
        assert len(obs1) == len(obs2) == len(acts) == len(rews) == len(done)
        self.len = len(obs1)
        self.obs1 = obs1
        self.obs2 = obs2
        self.acts = acts
        self.rews = rews
        self.done = done

    def __len__(self):
        return self.len


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return Batch(obs1=self.obs1_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     acts=self.acts_buf[idxs],
                     rews=self.rews_buf[idxs],
                     done=self.done_buf[idxs])


class NotEnoughDataInReplayBuffer(Exception):
    pass


class LockedReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, size):
        super().__init__(obs_dim, act_dim, size)
        self.lock = threading.Lock()

    def store(self, obs, act, rew=None, next_obs=None, done=None):
        with self.lock:
            super().store(obs, act, rew, next_obs, done)

    def sample_batch(self, batch_size=32):
        if self.size < batch_size:
            raise NotEnoughDataInReplayBuffer()
        with self.lock:
            return super().sample_batch(batch_size)


class TD3Policy(Policy):
    def __init__(self, name, env_id, obs_space, ac_space, n_envs, seed=0,
                 batch_size=256, batches_per_cycle=50, cycles_per_epoch=50, rollouts_per_worker=2,
                 gamma=0.99, polyak=0.999995, pi_lr=1e-3, q_lr=1e-3,
                 act_noise=0.1, target_noise=0.2, noise_clip=0.5, policy_delay=2,
                 noise_type='ou', noise_sigma=0.2,
                 n_initial_episodes=100, replay_size=int(1e6),
                 l2_coef=1e-4, train_mode=PolicyTrainMode.R_ONLY,
                 hidden_sizes=(256, 256, 256, 256),
                 sess_config=None, test_rollouts_per_epoch=10):
        assert policy_delay < batches_per_cycle
        assert noise_type in ['gaussian', 'ou']
        Policy.__init__(self, name, env_id, obs_space, ac_space, n_envs, seed)

        actor_critic = core.mlp_actor_critic
        ac_kwargs = dict(hidden_sizes=hidden_sizes)

        graph = tf.Graph()

        with graph.as_default():
            tf.set_random_seed(seed)
            np.random.seed(seed)

            obs_dim = obs_space.shape[0]
            act_dim = ac_space.shape[0]

            # Action limit for clamping: critically, assumes all dimensions share the same bound!
            act_limit = ac_space.high[0]

            # Share information about action space with policy architecture
            ac_kwargs['action_space'] = ac_space

            # Inputs to computation graph
            x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)
            bc_x_ph, bc_a_ph, _, _, _ = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

            pi, q1, q1_loss, q2, q2_loss, q_loss, td3_pi_loss = self.td3_graph(a_ph, ac_kwargs, act_limit, actor_critic,
                                                                               d_ph, gamma, noise_clip, r_ph,
                                                                               target_noise, x2_ph, x_ph)

            bc_pi_loss, l2_loss = self.bc_graph(ac_kwargs, act_dim, actor_critic, bc_a_ph, bc_x_ph, env_id, l2_coef, pi)

            td3_plus_bc_pi_loss = td3_pi_loss + bc_pi_loss

            # Separate train ops for pi, q
            pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
            train_pi_r_only_op = pi_optimizer.minimize(td3_pi_loss, var_list=get_vars('main/pi'))
            train_pi_bc_only_op = pi_optimizer.minimize(bc_pi_loss, var_list=get_vars('main/pi'))
            train_pi_td3_plus_bc_op = pi_optimizer.minimize(td3_plus_bc_pi_loss, var_list=get_vars('main/pi'))
            train_pi_ops = {
                PolicyTrainMode.R_ONLY: train_pi_r_only_op,
                PolicyTrainMode.SQIL_ONLY: train_pi_r_only_op,
                PolicyTrainMode.R_PLUS_SQIL: train_pi_r_only_op,
                PolicyTrainMode.BC_ONLY: train_pi_bc_only_op,
                PolicyTrainMode.R_PLUS_BC: train_pi_td3_plus_bc_op
            }
            q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
            train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

            # Polyak averaging for target variables
            target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                      for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

            # Initializing targets to match main variables
            target_init = tf.group([tf.assign(v_targ, v_main)
                                    for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

            if sess_config is None:
                sess_config = tf.ConfigProto()
                sess_config.gpu_options.allow_growth = True
            sess = tf.Session(config=sess_config, graph=graph)

            sess.run(tf.global_variables_initializer())
            sess.run(target_init)

        # Experience buffer
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        demonstrations_buffer = LockedReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        self.noise_sigma = np.ones((n_envs, act_dim))
        if isinstance(noise_sigma, float):
            self.noise_sigma *= noise_sigma
        elif isinstance(noise_sigma, np.ndarray):
            assert noise_sigma.shape == (act_dim,)
            self.noise_sigma *= noise_sigma
            """
            Yes, this does broadcast correctly:
                In [76]: a = np.ones((2, 3))
                In [77]: a * [2, 3, 5]
                Out[77]:
                array([[2., 3., 5.],
                       [2., 3., 5.]])
            """
        else:
            raise Exception()
        self.ou_noise = None
        self.obs_dim = obs_dim
        self.train_pi_op = train_pi_ops[train_mode]
        self.train_mode = train_mode
        self.train_pi_bc_only_op = train_pi_bc_only_op
        self.train_q_op = train_q_op
        self.target_update = target_update
        self.q1_loss = q1_loss
        self.q2_loss = q2_loss
        self.td3_pi_loss = td3_pi_loss
        self.bc_pi_loss = bc_pi_loss
        self.td3_plus_bc_pi_loss = td3_plus_bc_pi_loss
        self.q1 = q1
        self.q2 = q2
        self.q_loss = q_loss
        self.pi = pi
        self.act_limit = act_limit
        self.act_dim = act_dim
        self.act_noise = act_noise
        self.noise_type = noise_type
        self.x_ph = x_ph
        self.x2_ph = x2_ph
        self.a_ph = a_ph
        self.r_ph = r_ph
        self.d_ph = d_ph
        self.bc_x_ph = bc_x_ph
        self.bc_a_ph = bc_a_ph
        self.obs1 = None
        self.train_env = None
        self.test_env = None
        self.replay_buffer = replay_buffer
        self.demonstrations_buffer = demonstrations_buffer
        self.sess = sess
        self.batch_size = batch_size
        self.cycles_per_epoch = cycles_per_epoch
        self.batches_per_cycle = batches_per_cycle
        self.cycle_n = 1
        self.epoch_n = 1
        self.n_envs = n_envs
        self.initial_exploration_phase = True
        self.serial_episode_n = 0
        self.rollouts_per_worker = rollouts_per_worker
        self.policy_delay = policy_delay
        self.saver = None
        self.graph = graph
        self.n_initial_episodes = n_initial_episodes
        self.action_stats = LimitedRunningStat(shape=(act_dim,), len=1000)
        self.noise_stats = RunningStat(act_dim)
        self.ckpt_n = 0
        self.l2_loss = l2_loss
        self.seen_demonstrations = set()
        self.monitor_q_s = []
        self.monitor_q_a = []
        self.test_rollouts_per_epoch = test_rollouts_per_epoch
        self.last_test_obs = None
        self.reward_logger = None

        self.reset_noise()

    @staticmethod
    def bc_graph(ac_kwargs, act_dim, actor_critic, bc_a_ph, bc_x_ph, env_id, l2_coef, pi):
        # Behavioral cloning copy of main graph
        with tf.variable_scope('main', reuse=True):
            bc_pi, _, _, _ = actor_critic(bc_x_ph, bc_a_ph, **ac_kwargs)
            weights = [v for v in tf.trainable_variables() if '/kernel:0' in v.name]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])
        assert pi.shape.as_list() == bc_a_ph.shape.as_list()
        squared_differences = (bc_pi - bc_a_ph) ** 2
        assert squared_differences.shape.as_list() == [None, act_dim]
        if 'Fetch' in env_id:
            # Place more weight on gripper action
            squared_differences = tf.concat([squared_differences[:, :3], 10 * squared_differences[:, 3, None]],
                                            axis=1)
        assert squared_differences.shape.as_list() == [None, act_dim]
        squared_norms = tf.reduce_sum(squared_differences, axis=1)
        assert squared_norms.shape.as_list() == [None]
        bc_pi_loss = tf.reduce_mean(squared_norms, axis=0)
        assert bc_pi_loss.shape.as_list() == []
        bc_pi_loss += l2_coef * l2_loss
        return bc_pi_loss, l2_loss

    @staticmethod
    def td3_graph(a_ph, ac_kwargs, act_limit, actor_critic, d_ph, gamma, noise_clip, r_ph, target_noise, x2_ph, x_ph):
        # Main outputs from computation graph
        with tf.variable_scope('main'):
            pi, q1, q2, q1_pi = actor_critic(x_ph, a_ph, **ac_kwargs)
        # Target policy network
        with tf.variable_scope('target'):
            pi_targ, _, _, _ = actor_critic(x2_ph, a_ph, **ac_kwargs)
        # Target Q networks
        with tf.variable_scope('target', reuse=True):
            # Target policy smoothing, by adding clipped noise to target actions
            epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
            epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = tf.clip_by_value(a2, -act_limit, act_limit)

            # Target Q-values, using action from target policy
            _, q1_targ, q2_targ, _ = actor_critic(x2_ph, a2, **ac_kwargs)
        # Bellman backup for Q functions, using Clipped Double-Q targets
        min_q_targ = tf.minimum(q1_targ, q2_targ)
        backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * min_q_targ)
        # TD3 losses
        td3_pi_loss = -tf.reduce_mean(q1_pi)
        q1_loss = tf.reduce_mean((q1 - backup) ** 2)
        q2_loss = tf.reduce_mean((q2 - backup) ** 2)
        q_loss = q1_loss + q2_loss
        return pi, q1, q1_loss, q2, q2_loss, q_loss, td3_pi_loss

    def reset_noise(self):
        mu = np.zeros((self.n_envs, self.act_dim))
        self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=mu, sigma=self.noise_sigma)

    def test_agent(self):
        timer = TimerContext(name=None, stdout=False)
        print("Running test episodes...")
        # Logging will be taken care of by the environment itself (see env.py)
        # Why do the env reset in this funny way? Because we need to end with a reset to trigger
        # SaveEpisodeStats to save the stats from the final episode.
        # (FYI: test_env is a SubprocVecEnv - see env.py for the reason)
        rewards = []
        with timer:
            for _ in range(self.test_rollouts_per_epoch):
                obs, done = self.last_test_obs, False
                episode_reward = 0
                while not done:
                    [obs], [reward], [done], _ = self.test_env.step([self.step(obs, deterministic=True)])
                    episode_reward += reward
                rewards.append(episode_reward)
                self.last_test_obs = self.test_env.reset()[0]
        self.logger.logkv(f'policy_{self.name}/test_env_time_ms', timer.duration_s * 1000)
        return rewards

    def run_train_env_episode(self):
        timer = TimerContext(name=None, stdout=False)
        with timer:
            obses = self.obs1
            # Generate reset states for demonstrations
            # (We only care about the first environment, because that's the one from which reset states are generated)
            # TODO: we should be generating reset states from the test environment
            while True:
                obses, reward, dones, info = self.train_env.step(self.train_step(obses))
                for i in range(self.n_envs):
                    if dones[i]:
                        self.obs1[i] = self.train_env.reset_one_env(i)
                if dones[0]:
                    break
        if timer.duration_s is not None:
            self.logger.logkv(f'policy_{self.name}/train_env_time_ms', timer.duration_s * 1000)

    def train_bc_batch(self):
        if self.demonstrations_buffer.size <= self.batch_size:
            return None
        timer = TimerContext(name=None, stdout=False)
        with timer:
            loss_bc_pi_l, loss_l2_l = [], []
            for _ in range(self.batches_per_cycle):
                bc_batch = self.demonstrations_buffer.sample_batch(self.batch_size)
                feed_dict = {self.bc_x_ph: bc_batch.obs1, self.bc_a_ph: bc_batch.acts}
                bc_pi_loss, l2_loss, _ = self.sess.run([self.bc_pi_loss,
                                                        self.l2_loss,
                                                        self.train_pi_bc_only_op],
                                                       feed_dict)
                loss_bc_pi_l.append(bc_pi_loss)
                loss_l2_l.append(l2_loss)
            self.logger.log_list_stats(f'policy_{self.name}/loss_bc_pi', loss_bc_pi_l)
            self.logger.log_list_stats(f'policy_{self.name}/loss_l2', loss_l2_l)
            self.cycle_n += 1
            self.logger.logkv(f'policy_{self.name}/cycle', self.cycle_n)
        self.logger.logkv(f'policy_{self.name}/bc_train_time_ms', timer.duration_s * 1000)
        return np.mean(loss_bc_pi_l)

    def train_bc_only(self):
        # Takes about 200 ms
        bc_loss = self.train_bc_batch()

        # Takes about 7 seconds
        # 300: run about every 60 seconds
        if self.cycle_n % 300 == 0:
            self.run_train_env_episode()

        # Takes about 3 minutes (because saves videos and rendering is slooooow)
        # 5,000: run about every 15 minutes
        if self.cycle_n % 5000 == 0:
            self.epoch_n += 1
            self.logger.logkv(f'policy_{self.name}/epoch', self.epoch_n)
            self.test_agent()

        return bc_loss

    def process_rewards(self, obses, env_rewards):
        assert obses.shape == (self.train_env.num_envs,) + self.train_env.observation_space.shape
        assert env_rewards.shape == (self.train_env.num_envs,)
        reward_selector_rewards = global_variables.reward_selector.rewards(obses, env_rewards)
        return reward_selector_rewards

    def train(self):
        if self.train_env is None:
            raise Exception("env not set")

        if self.train_mode == PolicyTrainMode.NO_TRAINING:
            # Just run the environment to e.g. generate segments for DRLHP
            action = self.get_noise()
            _, _, dones, _ = self.train_env.step(action)
            for n, done in enumerate(dones):
                if done:
                    self.train_env.reset_one_env(n)
            return

        if self.train_mode == PolicyTrainMode.BC_ONLY:
            self.train_bc_only()
            return

        if self.initial_exploration_phase and self.serial_episode_n * self.n_envs >= self.n_initial_episodes:
            self.initial_exploration_phase = False
            print("Finished initial exploration at", str(datetime.datetime.now()))
            print("Size of replay buffer:", self.replay_buffer.size)

        if self.initial_exploration_phase:
            action = self.get_noise()
        else:
            action = self.train_step(self.obs1)

        # Step the env
        obs2, reward, done, _ = self.train_env.step(action)
        self.n_serial_steps += 1

        # Maybe replace rewards with e.g. predicted rewards
        reward = self.process_rewards(obs2, reward)
        self.reward_logger.log([reward], [done])

        # Store experience to replay buffer
        for i in range(self.n_envs):
            self.replay_buffer.store(self.obs1[i], action[i], reward[i], obs2[i], done[i])
            # So that obs1 is immediately set to the first obs from the next episode
            if done[i]:
                obs2[i] = self.train_env.reset_one_env(i)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        self.obs1 = obs2

        if done[0]:
            self.serial_episode_n += 1
            cycle_done = (self.serial_episode_n % self.rollouts_per_worker == 0)
        else:
            cycle_done = False

        if self.initial_exploration_phase:
            return

        if cycle_done:
            print(f"Cycle {self.cycle_n} done")

            self._train_rl()

            for n in range(self.act_dim):
                self.logger.logkv(f'policy_{self.name}/actions_mean_{n}', self.action_stats.mean[n])
                self.logger.logkv(f'policy_{self.name}/actions_std_{n}', self.action_stats.std[n])
                self.logger.logkv(f'policy_{self.name}/noise_mean_{n}', self.noise_stats.mean[n])
                self.logger.logkv(f'policy_{self.name}/noise_std_{n}', self.noise_stats.var[n])
            self.logger.logkv(f'policy_{self.name}/replay_buffer_ptr', self.replay_buffer.ptr)
            self.logger.logkv(f'policy_{self.name}/replay_buffer_demo_ptr', self.demonstrations_buffer.ptr)
            self.logger.logkv(f'policy_{self.name}/cycle', self.cycle_n)
            self.logger.logkv(f'policy_{self.name}/n_total_steps', self.n_total_steps())
            self.logger.measure_rate(f'policy_{self.name}/n_total_steps', self.n_total_steps(),
                                     f'policy_{self.name}/n_total_steps_per_second')

            if self.cycle_n and self.cycle_n % self.cycles_per_epoch == 0:
                self.epoch_n += 1
                self.logger.logkv(f'policy_{self.name}/epoch', self.epoch_n)
                self.test_agent()

            self.cycle_n += 1

    def _train_rl(self):
        results = defaultdict(list)
        for batch_n in range(self.batches_per_cycle):
            # Experience from normal replay buffer for regular Q-learning
            explore_batch = self.replay_buffer.sample_batch(self.batch_size)
            if self.train_mode == PolicyTrainMode.R_PLUS_SQIL:
                self.check_sqil_reward(explore_batch)
            if self.train_mode == PolicyTrainMode.SQIL_ONLY:
                explore_batch.rews = np.array([0] * self.batch_size)

            if self.train_mode in [PolicyTrainMode.SQIL_ONLY, PolicyTrainMode.R_PLUS_SQIL]:
                demo_batch = self.demonstrations_buffer.sample_batch(self.batch_size)
                demo_batch.rews = np.array([SQIL_REWARD] * self.batch_size)
                batch = combine_batches(explore_batch, demo_batch)
            else:
                batch = explore_batch

            feed_dict = {
                self.x_ph: batch.obs1, self.x2_ph: batch.obs2,
                self.a_ph: batch.acts, self.r_ph: batch.rews, self.d_ph: batch.done,
            }
            self.train_q(feed_dict, results)
            # Delayed policy update
            if batch_n % self.policy_delay == 0:
                self.train_pi(feed_dict, results)

        self.check_specific_states_qs()

        for k, l in results.items():
            self.logger.log_list_stats(f'policy_{self.name}/' + k, l)

    @staticmethod
    def check_sqil_reward(explore_batch):
        max_r = np.max(explore_batch.rews)
        if max_r >= SQIL_REWARD:
            print("Error: max. reward while exploring {:.3f} greater than SQIL reward".format(max_r))

    def check_specific_states_qs(self):
        if self.monitor_q_s:
            q1, q2, pi = self.sess.run([self.q1, self.q2, self.pi],
                                       feed_dict={self.x_ph: self.monitor_q_s,
                                                  self.a_ph: self.monitor_q_a})
            for i in range(len(q1)):
                self.logger.logkv(f'q_checks/q1_{i}', q1[i])
                self.logger.logkv(f'q_checks/q2_{i}', q2[i])
                self.logger.logkv(f'q_checks/pi_{i}', np.linalg.norm(pi - self.monitor_q_a[i]))

    def train_pi(self, feed_dict, results):
        fetches = {
            'loss_td3_pi': self.td3_pi_loss,
            'loss_l2': self.l2_loss,
        }
        # Behavioral cloning
        if self.train_mode == PolicyTrainMode.R_PLUS_BC:
            bc_batch = self.demonstrations_buffer.sample_batch(self.batch_size)
            feed_dict.update({self.bc_x_ph: bc_batch.obs1,
                              self.bc_a_ph: bc_batch.acts})
            fetches.update({'loss_bc_pi': self.bc_pi_loss,
                            'loss_td3_plus_bc_pi': self.td3_plus_bc_pi_loss})
        # train_pi_op is automatically set to be appropriate for the mode
        # (i.e. it /does/ do BC training if the policy was initialised with a BC mode)
        fetch_vals = self.sess.run(list(fetches.values()) + [self.train_pi_op, self.target_update],
                                   feed_dict)[:-2]
        self.update_results(fetch_vals, results, fetches)

    def train_q(self, feed_dict, results):
        fetches = {
            'loss_q': self.q_loss, 'loss_q1': self.q1_loss, 'loss_q2': self.q2_loss,
            'q1_vals': self.q1,
            'q2_vals': self.q2
        }
        fetch_vals = self.sess.run(list(fetches.values()) + [self.train_q_op], feed_dict)[:-1]
        self.update_results(fetch_vals, results, fetches)

    @staticmethod
    def update_results(fetch_vals, fetch_vals_l, fetches):
        for k, v in zip(fetches.keys(), fetch_vals):
            if isinstance(v, np.float32):
                fetch_vals_l[k].append(v)
            else:
                fetch_vals_l[k].extend(v)

    # Why use two functions rather than just having a 'deterministic' argument?
    # Because we need to be careful that the batch size matches the number of
    # environments for OU noise

    def get_noise(self):
        if self.noise_type == 'gaussian':
            noise = self.act_noise * np.random.randn(self.n_envs, self.act_dim)
        elif self.noise_type == 'ou':
            noise = self.ou_noise()
        else:
            raise Exception()
        assert noise.shape == (self.n_envs, self.act_dim)
        self.noise_stats.push(noise[0])
        return noise

    def train_step(self, o):
        assert o.shape == (self.n_envs, self.obs_dim)

        a = self.sess.run(self.pi, feed_dict={self.x_ph: o})
        assert a.shape == (self.n_envs, self.act_dim)

        noise = self.get_noise()
        assert noise.shape == (self.n_envs, self.act_dim)
        assert noise.shape == a.shape
        a += noise

        a = np.clip(a, -self.act_limit, self.act_limit)

        assert a.shape, (self.n_envs, self.act_dim)
        self.action_stats.push(a[0])
        self.actions_log_file.write(str(a[0]) + '\n')  # TODO debugging, deleteme

        return a

    def test_step(self, o):
        assert o.shape == (self.obs_dim,)
        a = self.sess.run(self.pi, feed_dict={self.x_ph: [o]})[0]
        return a

    def step(self, o, deterministic=True):
        # There are two reasons we might ever need to do a non-deterministic step:
        # - If we're training (but train() calls train_step directly)
        # - If we want a rollout with a bit of noise (which we shouldn't ever do for Fetch because we disable redo)
        assert deterministic
        return self.test_step(o)

    def make_saver(self):
        with self.graph.as_default():
            with self.sess.as_default():
                # var_list=tf.trainable_variables()
                # => don't try and load/save Adam variables
                self.saver = tf.train.Saver(max_to_keep=10,
                                            var_list=tf.trainable_variables())

    @staticmethod
    def second_newest_checkpoint(ckpt_prefix):
        ckpts = [f.replace('.index', '') for f in glob.glob(ckpt_prefix + '*.index')]
        # expects checkpoint names like network.ckpt-10
        ckpt = list(sorted(ckpts, key=lambda k: int(k.split('-')[-1])))[-2]
        return ckpt

    def load_checkpoint(self, path):
        if self.saver is None:
            self.make_saver()
        self.saver.restore(self.sess, path)
        print("Restored policy checkpoint from '{}'".format(path))

    def save_checkpoint(self, path):
        if self.saver is None:
            self.make_saver()
        saved_path = self.saver.save(self.sess, path, self.ckpt_n)
        self.ckpt_n += 1
        print("Saved policy checkpoint to '{}'".format(saved_path))

    def set_training_env(self, env, log_dir):
        self.train_env = env
        self.obs1 = np.array([self.train_env.reset_one_env(n)
                              for n in range(self.train_env.num_envs)])
        self.reward_logger = EpisodeRewardLogger(log_dir, n_steps=1, n_envs=self.n_envs)
        self.actions_log_file = open(os.path.join(log_dir, 'actions'), 'w')  # TODO debugging; deleteme

    def set_test_env(self, env: VecEnv, log_dir):
        assert env.unwrapped.num_envs == 1, env.unwrapped.num_envs
        self.test_env = env
        self.last_test_obs = self.test_env.reset()[0]

    def use_demonstrations(self, demonstrations: RolloutsByHash):
        def f():
            while True:
                for demonstration_hash in demonstrations.keys():
                    if demonstration_hash in self.seen_demonstrations:
                        continue
                    d = demonstrations[demonstration_hash]
                    assert len(d.obses) == len(d.actions)
                    o1s = d.obses[:-1]
                    acts = d.actions[:-1]
                    o2s = d.obses[1:]
                    dones = [0] * len(o1s)
                    assert len(o1s) == len(acts) == len(o2s) == len(dones), \
                        (len(o1s), len(acts), len(o2s), len(dones))
                    for o1, a, o2, done in zip(o1s, acts, o2s, dones):
                        self.demonstrations_buffer.store(obs=o1, act=a, next_obs=o2, done=done, rew=None)
                    self.seen_demonstrations.add(demonstration_hash)
                self.logger.logkv(f'policy_{self.name}/replay_buffer_demo_ptr', self.demonstrations_buffer.ptr)
                time.sleep(1)

        Thread(target=f).start()


def combine_batches(b1, b2):
    assert b1.len == b2.len
    return Batch(
        obs1=np.concatenate([b1.obs1, b2.obs1], axis=0),
        obs2=np.concatenate([b1.obs2, b2.obs2], axis=0),
        acts=np.concatenate([b1.acts, b2.acts], axis=0),
        rews=np.concatenate([b1.rews, b2.rews], axis=0),
        done=np.concatenate([b1.done, b2.done], axis=0),
    )
