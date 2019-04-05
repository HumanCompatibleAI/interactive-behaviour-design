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

from baselines.common.running_stat import RunningStat
from baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from policies.base_policy import Policy, PolicyTrainMode
from rollouts import RolloutsByHash

SQIL_REWARD = 5


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
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class LockedReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, size):
        super().__init__(obs_dim, act_dim, size)
        self.lock = threading.Lock()

    def store(self, obs, act, rew=None, next_obs=None, done=None):
        with self.lock:
            super().store(obs, act, rew, next_obs, done)

    def sample_batch(self, batch_size=32):
        while self.size == 0:
            print("Warning: demonstrations buffer empty; waiting...")
            time.sleep(1)
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
                 sess_config=None):
        assert policy_delay < batches_per_cycle
        assert noise_type in ['gaussian', 'ou']
        Policy.__init__(self, name, env_id, obs_space, ac_space, n_envs, seed)

        actor_critic = core.mlp_actor_critic
        ac_kwargs = dict(hidden_sizes=hidden_sizes)
        self.n_serial_steps = 0

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

            # Main outputs from computation graph
            with tf.variable_scope('main'):
                pi, q1, q2, q1_pi = actor_critic(x_ph, a_ph, **ac_kwargs)

            # Behavioral cloning copy of main graph
            with tf.variable_scope('main', reuse=True):
                bc_pi, _, _, _ = actor_critic(bc_x_ph, bc_a_ph, **ac_kwargs)
                weights = [v for v in tf.trainable_variables() if '/kernel:0' in v.name]
                l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])

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

            # Experience buffer
            replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
            demonstrations_buffer = LockedReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

            # Bellman backup for Q functions, using Clipped Double-Q targets
            min_q_targ = tf.minimum(q1_targ, q2_targ)
            backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * min_q_targ)

            # TD3 losses
            td3_pi_loss = -tf.reduce_mean(q1_pi)
            q1_loss = tf.reduce_mean((q1 - backup) ** 2)
            q2_loss = tf.reduce_mean((q2 - backup) ** 2)
            q_loss = q1_loss + q2_loss

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

            td3_plus_bc_pi_loss = td3_pi_loss + bc_pi_loss

            # Separate train ops for pi, q
            pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
            q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
            train_pi_td3_only_op = pi_optimizer.minimize(td3_pi_loss, var_list=get_vars('main/pi'))
            train_pi_bc_only_op = pi_optimizer.minimize(bc_pi_loss, var_list=get_vars('main/pi'))
            train_pi_td3_plus_bc_op = pi_optimizer.minimize(td3_plus_bc_pi_loss, var_list=get_vars('main/pi'))
            train_pi_ops = {
                PolicyTrainMode.R_ONLY: train_pi_td3_only_op,
                PolicyTrainMode.SQIL_ONLY: train_pi_td3_only_op,
                PolicyTrainMode.R_PLUS_SQIL: train_pi_td3_only_op,
                PolicyTrainMode.BC_ONLY: train_pi_bc_only_op,
                PolicyTrainMode.R_PLUS_BC: train_pi_td3_plus_bc_op
            }
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
        self.env = None
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
        self.action_stats = RunningStat(act_dim)
        self.noise_stats = RunningStat(act_dim)
        self.ckpt_n = 0
        self.l2_loss = l2_loss
        self.seen_demonstrations = set()

        self.reset_noise()

    def reset_noise(self):
        mu = np.zeros((self.n_envs, self.act_dim))
        self.ou_noise = OrnsteinUhlenbeckActionNoise(mu=mu, sigma=self.noise_sigma)

    def test_agent(self, n=10):
        rets = []
        for j in range(n):
            o, r, d, ep_ret, ep_len = self.test_env.reset(), 0, False, 0, 0
            while not d:
                # Take deterministic actions at test time
                o, r, d, _ = self.test_env.step(self.test_step(o))
                ep_ret += r
                ep_len += 1
            rets.append(ep_ret)
            self.logger.logkv('test_ep_reward', ep_ret)
            self.logger.logkv('test_ep_len', ep_len)
        return rets

    def train_bc(self):
        loss_bc_pi_l, loss_l2_l = [], []
        for _ in range(self.batches_per_cycle):
            bc_batch = self.demonstrations_buffer.sample_batch(self.batch_size)
            feed_dict = {
                self.bc_x_ph: bc_batch['obs1'], self.bc_a_ph: bc_batch['acts']
            }
            bc_pi_loss, l2_loss, _ = self.sess.run([self.bc_pi_loss, self.l2_loss, self.train_pi_bc_only_op], feed_dict)
            loss_bc_pi_l.append(bc_pi_loss)
            loss_l2_l.append(l2_loss)
        self.logger.log_list_stats(f'policy_{self.name}/loss_bc_pi', loss_bc_pi_l)
        self.logger.log_list_stats(f'policy_{self.name}/loss_l2', loss_l2_l)
        self.cycle_n += 1
        self.logger.logkv(f'policy_{self.name}/cycle', self.cycle_n)
        if self.cycle_n % self.cycles_per_epoch == 0:
            self.epoch_n += 1
            self.logger.logkv(f'policy_{self.name}/epoch', self.epoch_n)
        return np.mean(loss_bc_pi_l)

    def train(self):
        if self.train_mode == PolicyTrainMode.BC_ONLY:
            self.train_bc()
            return

        if self.env is None:
            raise Exception("env not set")

        if self.initial_exploration_phase and self.serial_episode_n * self.n_envs >= self.n_initial_episodes:
            self.initial_exploration_phase = False
            print("Finished initial exploration at", str(datetime.datetime.now()))
            print("Size of replay buffer:", self.replay_buffer.size)

        if self.initial_exploration_phase:
            action = self.get_noise()
        else:
            action = self.train_step(self.obs1)

        # Step the env
        obs2, reward, done, _ = self.env.step(action)
        self.n_serial_steps += 1

        # Store experience to replay buffer
        for i in range(self.n_envs):
            self.replay_buffer.store(self.obs1[i], action[i], reward[i], obs2[i], done[i])
            if done[i]:
                obs2[i] = self.env.reset_one_env(i)

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
            n_total_steps = self.n_serial_steps * self.n_envs
            self.logger.logkv(f'policy_{self.name}/n_total_steps', n_total_steps)
            self.logger.measure_rate(f'policy_{self.name}/n_total_steps', n_total_steps,
                                     f'policy_{self.name}/n_total_steps_per_second')

            if self.cycle_n and self.cycle_n % self.cycles_per_epoch == 0:
                self.epoch_n += 1
                self.logger.logkv(f'policy_{self.name}/epoch', self.epoch_n)

            self.cycle_n += 1

    @staticmethod
    def combine_batches(b1, b2):
        assert len(list(b1.keys())) == len(list(b2.keys()))
        for k in b1.keys():
            assert b1[k].shape == b2[k].shape
            assert isinstance(b1[k], np.ndarray)
            assert isinstance(b2[k], np.ndarray)

        b = {}
        b['obs1'] = np.concatenate([b1['obs1'], b2['obs1']], axis=0)
        b['obs2'] = np.concatenate([b1['obs2'], b2['obs2']], axis=0)
        b['acts'] = np.concatenate([b1['acts'], b2['acts']], axis=0)
        b['rews'] = np.concatenate([b1['rews'], b2['rews']], axis=0)
        b['done'] = np.concatenate([b1['done'], b2['done']], axis=0)

        for k in b1.keys():
            expected_shape = (b1[k].shape[0] + b2[k].shape[0],) + b1[k].shape[1:]
            assert b[k].shape == expected_shape

        return b

    def _train_rl(self):
        fetch_vals_l = defaultdict(list)
        for batch_n in range(self.batches_per_cycle):
            # Experience from normal replay buffer for regular Q-learning
            explore_batch = self.replay_buffer.sample_batch(self.batch_size)
            if self.train_mode == PolicyTrainMode.R_PLUS_SQIL:
                max_r = np.max(explore_batch['rews'])
                if max_r >= SQIL_REWARD:
                    print("(Potential) error: max. reward while exploring {:.3f}is greater than SQIL reward".format(max_r))
            if self.train_mode in [PolicyTrainMode.SQIL_ONLY, PolicyTrainMode.R_PLUS_SQIL]:
                demo_batch = self.demonstrations_buffer.sample_batch(self.batch_size)
                if self.train_mode == PolicyTrainMode.SQIL_ONLY:
                    explore_batch['rews'] = np.array([0] * self.batch_size)
                demo_batch['rews'] = np.array([SQIL_REWARD] * self.batch_size)
                batch = self.combine_batches(explore_batch, demo_batch)
            else:
                batch = explore_batch
            feed_dict = {
                self.x_ph: batch['obs1'],
                self.x2_ph: batch['obs2'],
                self.a_ph: batch['acts'],
                self.r_ph: batch['rews'],
                self.d_ph: batch['done'],
            }

            fetches = {
                'loss_q': self.q_loss,
                'loss_q1': self.q1_loss,
                'loss_q2': self.q2_loss,
                'q1_vals': self.q1,
                'q2_vals': self.q2
            }
            fetch_vals = self.sess.run(list(fetches.values()) + [self.train_q_op], feed_dict)[:-1]
            for k, v in zip(fetches.keys(), fetch_vals):
                if isinstance(v, np.float32):
                    fetch_vals_l[k].append(v)
                else:
                    fetch_vals_l[k].extend(v)

            # Delayed policy update
            if batch_n % self.policy_delay == 0:
                fetches = {
                    'loss_td3_pi': self.td3_pi_loss,
                    'loss_l2': self.l2_loss,
                }

                # Behavioral cloning
                if self.train_mode == PolicyTrainMode.R_PLUS_BC:
                    bc_batch = self.demonstrations_buffer.sample_batch(self.batch_size)
                    feed_dict.update({self.bc_x_ph: bc_batch['obs1'],
                                      self.bc_a_ph: bc_batch['acts']})
                    fetches.update({'loss_bc_pi': self.bc_pi_loss,
                                    'loss_td3_plus_bc_pi': self.td3_plus_bc_pi_loss})

                # train_pi_op is automatically set to be appropriate for the mode
                # (i.e. it /does/ do BC training if the policy was initialised with a BC mode)
                fetch_vals = self.sess.run(list(fetches.values()) + [self.train_pi_op, self.target_update],
                                           feed_dict)[:-2]

                for k, v in zip(fetches.keys(), fetch_vals):
                    if isinstance(v, np.float32):
                        fetch_vals_l[k].append(v)
                    else:
                        fetch_vals_l[k].extend(v)
        for k, l in fetch_vals_l.items():
            self.logger.log_list_stats(f'policy_{self.name}/' + k, l)

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

    def set_training_env(self, env):
        self.env = env
        self.obs1 = self.env.reset()

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
                    dones = [0] * (len(d.obses) - 1) + [1]
                    assert len(o1s) == len(acts) == len(o2s) == len(dones)
                    for o, a, o2, done in zip(o1s, acts, o2s, dones):
                        self.demonstrations_buffer.store(obs=o, act=a, next_obs=o2, done=done, rew=None)
                    self.seen_demonstrations.add(demonstration_hash)
                self.logger.logkv(f'policy_{self.name}/replay_buffer_demo_ptr', self.demonstrations_buffer.ptr)
                time.sleep(1)

        Thread(target=f).start()
