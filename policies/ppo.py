import hashlib
import random
import time
from collections.__init__ import namedtuple

import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete

import global_variables
from baselines.common.policies import build_policy
from baselines.ppo2.ppo2 import constfn, Model as PPOModel, Runner as PPORunner
from policies.base_policy import Policy, PolicyTrainMode
from utils import RateMeasure, batch_iter, LogTime, Timer, sample_demonstration_batch, NotEnoughDemonstrations


class PPOPolicy(Policy):
    @staticmethod
    def get_hyperparams(env_type):
        if env_type == 'mujoco':
            # from ppo2/defaults.py
            return dict(
                nsteps=2048, nminibatches=32,
                lam=0.95, gamma=0.99, noptepochs=10,
                log_interval=1,
                ent_coef=0.0,
                lr=lambda f: 3e-4 * f,
                cliprange=constfn(0.2),
                network='mlp',
                network_args=dict(value_network='copy'),
            )
        elif env_type == 'atari':
            # from ppo2/defaults.py
            return dict(
                nsteps=128, nminibatches=4,
                lam=0.95, gamma=0.99, noptepochs=4,
                log_interval=1,
                ent_coef=.01,
                lr=lambda f: f * 2.5e-4,
                cliprange=lambda f: f * 0.1,
                network='cnn',
                network_args=dict()
            )
        elif env_type == 'fetch':
            return dict(
                # From PPO paper results for roboschool
                nsteps=512, nminibatches=4,
                lam=0.95, gamma=0.99, noptepochs=15,
                log_interval=10,
                ent_coef=0.0,
                lr=constfn(3e-4),
                cliprange=constfn(0.2),
                network='mlp',
                network_args=dict()
            )
        elif env_type == 'lunarlander':
            return dict(
                nsteps=128, nminibatches=4,
                lam=0.95, gamma=0.99, noptepochs=4,
                log_interval=10,
                ent_coef=0.0,
                lr=constfn(3e-4),
                cliprange=constfn(0.2),
                network='mlp',
                network_args=dict()
            )
        else:
            # from defaults in ppo2/ppo2.py
            return dict(
                nsteps=2048, nminibatches=4,
                lam=0.95, gamma=0.99, noptepochs=4,
                log_interval=10,
                ent_coef=0.0,
                lr=constfn(3e-4),
                cliprange=constfn(0.2),
                network='mlp',
                network_args=dict()
            )

    def __init__(self, name, env_id, obs_space, ac_space, n_envs, seed=None):
        Policy.__init__(self, name, env_id, obs_space, ac_space, n_envs)

        atari_envs = ['Seaquest', 'Breakout', 'SpaceInvaders', 'BeamRider', 'Qbert', 'Enduro', 'Pong']
        if any([e in env_id for e in atari_envs]):
            hyperparams = PPOPolicy.get_hyperparams('atari')
        elif 'Fetch' in env_id:
            hyperparams = PPOPolicy.get_hyperparams('fetch')
        elif 'LunarLander' in env_id:
            hyperparams = PPOPolicy.get_hyperparams('lunarlander')
        elif 'Reacher' in env_id or 'Ant' in env_id:
            hyperparams = PPOPolicy.get_hyperparams('default')
        else:
            raise RuntimeError(f"Unsure of PPO hyperparameters for '{env_id}'")

        nbatch = n_envs * hyperparams['nsteps']
        nbatch_train = nbatch // hyperparams['nminibatches']

        T = namedtuple('env', ['action_space', 'observation_space'])
        env_shim = T(ac_space, obs_space)
        graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=graph, config=config)
        with graph.as_default():
            if seed is None:
                seed = int(hashlib.sha1(name.encode()).hexdigest()[:4], base=16)
            tf.set_random_seed(seed)
            with sess.as_default():
                tf.set_random_seed(seed)
                policy = build_policy(env_shim,
                                      hyperparams['network'],
                                      **hyperparams['network_args'])
                model = PPOModel(policy=policy,
                                 ob_space=obs_space,
                                 ac_space=ac_space,
                                 nbatch_act=n_envs,
                                 nbatch_train=nbatch_train,
                                 nsteps=hyperparams['nsteps'],
                                 ent_coef=hyperparams['ent_coef'],
                                 vf_coef=0.5,
                                 max_grad_norm=0.5)

        self.graph = graph
        self.sess = sess
        self.model = model
        self.nbatch = nbatch
        self.noptepochs = hyperparams['noptepochs']
        self.nbatch_train = nbatch_train
        self.lr = hyperparams['lr']
        self.cliprange = hyperparams['cliprange']
        self.nsteps = hyperparams['nsteps']
        self.gamma = hyperparams['gamma']
        self.lam = hyperparams['lam']
        self.step_measure = RateMeasure()
        self.step_measure.reset(0)
        self.log_interval = 1
        self.runner = None
        self.saver = None
        self.training_enabled = False
        self.n_envs = n_envs
        self.action_space = ac_space
        self.obs_space = obs_space
        self.n_bc_epochs = 0
        # During initial testing, it took about 2 seconds to run one episode of Fetch
        # The idea is: train BC for about 5 times as long as we run the testing environment
        self.train_bc_time = 10
        self.last_obs = None

    def step(self, obs, obs_is_batch=False, **step_kwargs):
        if isinstance(self.action_space, Discrete):
            n_actions = self.action_space.n
        elif isinstance(self.action_space, Box):
            assert len(self.action_space.shape) == 1
            n_actions = self.action_space.shape[0]
        else:
            raise Exception("Unsure how to determine no. actions for", self.action_space)

        batch_size = self.model.act_model.X.shape.as_list()[0]
        obs = np.array(obs)  # Could be LazyFrames
        if not obs_is_batch:
            obs = np.array([obs] * batch_size)
        assert obs.shape == (batch_size,) + self.obs_space.shape, (obs.shape, batch_size, self.obs_space.shape)

        if 'deterministic' in step_kwargs and step_kwargs['deterministic']:
            x = self.sess.run(self.model.act_model.pi, feed_dict={self.model.act_model.X: obs})
            assert x.shape == (batch_size, n_actions)
            if isinstance(self.action_space, Box):
                mean_action = x
                actions = mean_action
            elif isinstance(self.action_space, Discrete):
                logits = x
                actions = np.argmax(logits, axis=1)
            else:
                raise Exception("Unsure how to get noise-free action for", self.action_space)
        else:
            actions, _, _, _ = self.model.act_model.step(obs, **step_kwargs)
            if isinstance(self.action_space, Discrete):
                assert actions.shape == (batch_size,), actions.shape
            elif isinstance(self.action_space, Box):
                assert actions.shape == (batch_size, n_actions), actions.shape

        if obs_is_batch:
            return actions
        else:
            return actions[0]

    def set_training_env(self, env, log_dir):
        self.env = env
        if self.runner is None:
            self.runner = PPORunner(env=env,
                                    model=self.model,
                                    nsteps=self.nsteps,
                                    gamma=self.gamma,
                                    lam=self.lam,
                                    log_dir=log_dir)
            self.last_obs = self.runner.obs

    def set_test_env(self, env, log_dir):
        # We don't use a test environment; we just rely on the training environment
        pass

    def train(self):
        train_mode = self.train_mode  # Could be changed while we're running

        if train_mode == PolicyTrainMode.BC_ONLY:
            self.train_bc_only()
            return

        obs, returns, masks, actions, values, neglogpacs, states, epinfos = self.runner.run()

        if train_mode == PolicyTrainMode.NO_TRAINING:
            return  # just run the environment to e.g. generate segments for DRLHP

        assert train_mode in [PolicyTrainMode.R_PLUS_BC, PolicyTrainMode.R_ONLY], train_mode

        train_kwargs = {}
        train_kwargs['train_mode'] = train_mode

        if train_mode == PolicyTrainMode.R_PLUS_BC:
            try:
                bc_obs, bc_actions = sample_demonstration_batch(self.demonstration_rollouts)
            except NotEnoughDemonstrations as e:
                print(e)
                time.sleep(1.0)
                return
            train_kwargs.update({'bc_obses': bc_obs, 'bc_actions': bc_actions, 'train_mode': train_mode})
        lrnow = self.lr(1.0)
        cliprangenow = self.cliprange(1.0)

        inds = np.arange(self.nbatch)
        stats = []
        stats_keys = None
        for _ in range(self.noptepochs):
            np.random.shuffle(inds)
            for start in range(0, self.nbatch, self.nbatch_train):
                end = start + self.nbatch_train
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                fetches = self.model.train_rl_bc(lrnow, cliprangenow, *slices, **train_kwargs)
                stats_keys = list(fetches.keys())
                stats.append(list(fetches.values()))
        self.n_updates += 1
        self.n_serial_steps += self.runner.nsteps

        assert stats_keys is not None
        stats = np.array(stats)
        assert stats.shape == (self.noptepochs * self.nbatch // self.nbatch_train, len(stats_keys))
        stats = np.mean(stats, axis=0)

        if self.n_updates and self.n_updates % self.log_interval == 0:
            steps_per_second = self.step_measure.measure(self.n_total_steps)
            self.logger.logkv('policy_{}/n_updates'.format(self.name), self.n_updates)
            self.logger.logkv('policy_{}/n_serial_steps'.format(self.name), self.n_serial_steps)
            self.logger.logkv('policy_{}/n_total_steps'.format(self.name), self.n_total_steps())
            self.logger.logkv('policy_{}/n_total_steps_per_second'.format(self.name), steps_per_second)
            for n, stat_key in enumerate(stats_keys):
                self.logger.logkv('policy_{}/{}'.format(self.name, stat_key), stats[n])

    def train_bc_only(self):
        with LogTime('bc_training', self.logger):
            train_timer = Timer(duration_seconds=self.train_bc_time)
            train_timer.reset()
            while not train_timer.done():
                # Train one epoch over all demonstrations
                rollouts = self.demonstration_rollouts.values()
                obses = [obs for rollout in rollouts for obs in rollout.obses]
                actions = [a for rollout in rollouts for a in rollout.actions]
                self.logger.logkv(f'policy_{self.name}/n_demonstration_steps', len(obses))
                bc_losses = []
                l2_losses = []
                for batch in batch_iter(list(zip(obses, actions)), batch_size=256, shuffle=True):
                    obses_batch, actions_batch = zip(*batch)
                    bc_loss, l2_loss = self.train_bc(obses_batch, actions_batch)
                    bc_losses.append(bc_loss)
                    l2_losses.append(l2_loss)
                if bc_losses:
                    self.logger.logkv(f'policy_{self.name}/bc_loss_train_mean', np.mean(bc_losses))
                    self.logger.logkv(f'policy_{self.name}/bc_loss_train_max', np.max(bc_losses))
                    self.logger.logkv(f'policy_{self.name}/l2_loss', np.mean(l2_losses))
                self.n_bc_epochs += 1
                self.logger.logkv(f'policy_{self.name}/n_bc_epochs', self.n_bc_epochs)

        with LogTime('run_episode', self.logger):
            t1 = time.time()
            done = [False]
            obs = self.last_obs
            # Generate reset states for demonstrations
            # (We only care about the first environment, because that's the one from which reset states are generated)
            while not done[0]:
                a = self.step(obs, deterministic=True, obs_is_batch=True)
                obs, reward, done, info = self.env.step(a)
            self.last_obs = self.env.reset()
            t2 = time.time()
            self.train_bc_time = (t2 - t1) * 5
            return

    def train_bc(self, obs, actions):
        bc_loss, l2_loss = self.model.train_bc_only(obs, actions, lr=self.lr(1.0))
        return bc_loss, l2_loss

    def make_saver(self):
        with self.graph.as_default():
            with self.sess.as_default():
                # var_list=tf.trainable_variables()
                # => don't try and load/save Adam variables
                self.saver = tf.train.Saver(max_to_keep=2,
                                            var_list=tf.trainable_variables())

    def save_checkpoint(self, path):
        if self.saver is None:
            self.make_saver()
        saved_path = self.saver.save(self.sess, path)
        print("Saved policy checkpoint to '{}'".format(saved_path))

    def load_checkpoint(self, path):
        if self.saver is None:
            self.make_saver()
        self.saver.restore(self.sess, path)
        print("Restored policy checkpoint from '{}'".format(path))

    def use_demonstrations(self, demonstrations):
        self.demonstration_rollouts = demonstrations
