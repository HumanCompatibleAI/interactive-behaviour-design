import contextlib
import glob
import logging
import os
import os.path as osp
from enum import Enum

import easy_tf_log
import joblib
import numpy as np
import tensorflow as tf
from numpy.testing import assert_equal

import global_variables
import throttler
from drlhp.drlhp_utils import LimitedRunningStat, RunningStat
from drlhp.pref_db import PrefDB
from drlhp.reward_predictor_core_network import net_cnn
from utils import batch_iter, RateMeasure, LogMilliseconds

MIN_L2_REG_COEF = 0.001


class PredictedRewardNormalization(Enum):
    OFF = 0
    RUNNING_STATS = 1
    EXTREME_TRAINING_STATES = 2
    NORM_RANDOM_STATES = 3
    NORM_TRAINING_STATES = 5
    MANUAL = 4


class RewardPredictor:

    def __init__(self, obs_space, network, network_args, r_std, name, reward_normalization,
                 normalization_loss_coef, lr=1e-4, log_dir=None, seed=None, gpu_n=None):
        self.min_reward_obs_so_far = None
        self.max_reward_obs_so_far = None
        self.obs_space = obs_space
        self.obs_shape = obs_space.shape
        self.reward_normalization = reward_normalization
        graph = tf.Graph()
        self.graph = graph
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)

        if gpu_n is None:
            device_context = contextlib.suppress()
        else:
            device_context = graph.device(f'/gpu:{gpu_n}')

        with graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self.l2_reg_coef = MIN_L2_REG_COEF * 1000
            with device_context:
                self.rps = [RewardPredictorNetwork(core_network=network,
                                                   network_args=network_args,
                                                   obs_shape=obs_space.shape,
                                                   lr=lr,
                                                   reward_normalization=reward_normalization,
                                                   normalization_loss_coef=normalization_loss_coef)]
                self.init_op = tf.global_variables_initializer()
            self.summaries = self.add_summary_ops()

        self.train_writer = tf.summary.FileWriter(
            osp.join(log_dir, f'reward_predictor_{name}', 'train'), flush_secs=5)
        self.test_writer = tf.summary.FileWriter(
            osp.join(log_dir, f'reward_predictor_{name}', 'test'), flush_secs=5)

        self.n_steps = 0
        self.n_epochs = 0
        self.r_norm_limited = LimitedRunningStat()
        self.r_norm = RunningStat(shape=[])
        self.r_std = r_std

        self.logger = easy_tf_log.Logger()
        self.logger.set_log_dir(osp.join(log_dir, f'reward_predictor_{name}', 'misc'))
        self.log_n_epochs()
        self.reward_call_n = 0

        self.log_interval = 20

        self.ckpt_n = 0
        self.polyak_min = 1

        self.step_rate = RateMeasure()
        self.step_rate.reset(self.n_steps)

        self.init_network()

        self.restore_placeholders = None
        self.restore_ops = None
        self.polyak_restore_ops = None
        self.set_up_restore_ops()

    def set_up_restore_ops(self):
        with self.graph.as_default():
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            self.restore_placeholders = {}
            for var in variables:
                ph = tf.placeholder(tf.float32, var.shape)
                self.restore_placeholders[var.name] = ph

            self.restore_ops = []
            self.polyak_restore_ops = []
            self.polyak_ph = tf.placeholder(tf.float32)
            for var in variables:
                new_value = self.restore_placeholders[var.name]
                self.restore_ops.append(var.assign(new_value))
                polyak_new_value = self.polyak_ph * var + (1 - self.polyak_ph) * new_value
                self.polyak_restore_ops.append(var.assign(polyak_new_value))

    def add_summary_ops(self):
        summary_ops = []

        for pred_n, rp in enumerate(self.rps):
            name = 'reward_predictor/accuracy_{}'.format(pred_n)
            op = tf.summary.scalar(name, rp.accuracy)
            summary_ops.append(op)
            name = 'reward_predictor/loss_{}'.format(pred_n)
            op = tf.summary.scalar(name, rp.loss)
            summary_ops.append(op)
            name = 'reward_predictor/prediction_loss_{}'.format(pred_n)
            op = tf.summary.scalar(name, rp.prediction_loss)
            summary_ops.append(op)
            l2_reg_losses = [rp.l2_reg_loss for rp in self.rps]
            mean_reg_loss = tf.reduce_mean(l2_reg_losses)
            op = tf.summary.scalar('reward_predictor/l2_loss_mean', mean_reg_loss)
            summary_ops.append(op)

        summaries = tf.summary.merge(summary_ops)

        return summaries

    def init_network(self):
        self.sess.run(self.init_op)

    def save(self, path, max_to_keep=None):
        save_path = f"{path}.{self.ckpt_n}"

        variable_value_dict = self.get_variable_value_dict()
        save_dict = {variable.name: value for variable, value in variable_value_dict.items()}
        joblib.dump(save_dict, save_path)

        print("Saved reward predictor checkpoint to '{}'".format(save_path))
        self.ckpt_n += 1

        checkpoint_paths = glob.glob(path + '.*')
        checkpoint_paths.sort(key=lambda p: os.path.getmtime(p))
        if max_to_keep is not None:
            for path in checkpoint_paths[:-max_to_keep]:
                os.remove(path)

    def load(self, ckpt_path, polyak_coef=None):
        variables_values_dict = joblib.load(ckpt_path)
        assert len(variables_values_dict) == len(self.restore_placeholders)
        feed_dict = {self.restore_placeholders[var_name]: variables_values_dict[var_name]
                     for var_name in variables_values_dict.keys()}
        if polyak_coef is None:
            op = self.restore_ops
        else:
            op = self.polyak_restore_ops
            feed_dict[self.polyak_ph] = polyak_coef
        self.sess.run(op, feed_dict)
        print("Restored reward predictor from checkpoint '{}'".format(ckpt_path))

    @staticmethod
    def get_latest_checkpoint(path):
        checkpoints = glob.glob(path + '.*')
        checkpoints.sort(key=lambda p: os.path.getmtime(p))
        latest_checkpoint_path = checkpoints[-1]
        return latest_checkpoint_path

    def get_variable_value_dict(self):
        with self.graph.as_default():
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assert len(variables) > 0
        variable_values = self.sess.run(variables)
        return {variable: value
                for variable, value in zip(variables, variable_values)}

    def unnormalized_rewards(self, obs):
        """
        Return (unnormalized) reward for each frame of a single segment
        from each member of the ensemble.
        """
        assert_equal(obs.shape[1:], self.obs_shape)
        n_steps = obs.shape[0]
        feed_dict = self.get_feed_dict()
        for rp in self.rps:
            feed_dict[rp.training] = False
            feed_dict[rp.s1] = [obs]
        # This will return nested lists of sizes n_preds x 1 x nsteps
        # (x 1 because of the batch size of 1)
        with LogMilliseconds('instrumentation/reward_prediction', self.logger, log_every=1000):
            rs = self.sess.run([rp.r1 for rp in self.rps], feed_dict)
        rs = np.array(rs)
        # Get rid of the extra x 1 dimension
        rs = rs[:, 0, :]
        n_preds = 1
        assert_equal(rs.shape, (n_preds, n_steps))
        return rs

    def running_stats_normalize(self, rewards, update_normalization):
        rewards = np.copy(rewards)
        if update_normalization:
            for reward in rewards:
                self.r_norm_limited.push(reward)
                self.r_norm.push(reward)

        if self.reward_call_n % global_variables.log_reward_normalisation_every_n_calls == 0:
            self.logger.logkv('reward_predictor/r_norm_mean_recent', self.r_norm_limited.mean)
            self.logger.logkv('reward_predictor/r_norm_std_recent', self.r_norm_limited.std)
            self.logger.logkv('reward_predictor/r_norm_mean', self.r_norm.mean)
            self.logger.logkv('reward_predictor/r_norm_std', self.r_norm.std)

        rewards -= self.r_norm.mean
        rewards /= (self.r_norm.std + 1e-12)

        return rewards

    def extreme_state_normalize(self, obses, rewards, update_normalization):
        rewards = np.copy(rewards)
        if not update_normalization and self.max_reward_obs_so_far is None:
            return rewards

        if self.max_reward_obs_so_far is not None:
            max_reward, min_reward = self.unnormalized_rewards(np.array([self.max_reward_obs_so_far,
                                                                         self.min_reward_obs_so_far]))[0]
        else:
            max_reward = float('-inf')
            min_reward = float('inf')

        if np.max(rewards) > max_reward:
            if update_normalization:
                self.max_reward_obs_so_far = obses[np.argmax(rewards)]
            max_reward = np.max(rewards)
        if np.min(rewards) < min_reward:
            if update_normalization:
                self.min_reward_obs_so_far = obses[np.argmin(rewards)]
            min_reward = np.min(rewards)

        scale = max_reward - min_reward
        shift = (min_reward + max_reward) / 2

        if self.reward_call_n % global_variables.log_reward_normalisation_every_n_calls == 0:
            self.logger.logkv('reward_predictor/reward_cur_batch_min', np.min(rewards))
            self.logger.logkv('reward_predictor/reward_cur_batch_max', np.max(rewards))
            self.logger.logkv('reward_predictor/reward_max', max_reward)
            self.logger.logkv('reward_predictor/reward_min', min_reward)
            self.logger.logkv('reward_predictor/scale', scale)
            self.logger.logkv('reward_predictor/shift', shift)

        rewards /= scale
        rewards -= shift

    @staticmethod
    def manual_normalize(rewards):
        rewards = np.copy(rewards)
        assert global_variables.predicted_rewards_normalize_mean_std is not None, \
            "--predicted_rewards_normalize_params not specified"
        scale, shift = map(float, global_variables.predicted_rewards_normalize_mean_std.split(','))
        rewards *= scale
        rewards += shift
        return rewards

    def normalized_rewards(self, obses, update_normalization=True):
        assert_equal(obses.shape[1:], self.obs_shape)
        n_steps = obses.shape[0]

        ensemble_rs = self.unnormalized_rewards(obses)
        logging.debug("Unnormalized rewards:\n%s", ensemble_rs)

        # Normalize rewards

        n_preds = 1
        assert_equal(ensemble_rs.shape, (n_preds, n_steps))
        rewards = ensemble_rs[0, :]

        with LogMilliseconds('instrumentation/reward_normalization', self.logger, log_every=1000):
            if self.reward_normalization == PredictedRewardNormalization.OFF:
                pass
            elif self.reward_normalization == PredictedRewardNormalization.RUNNING_STATS:
                rewards = self.running_stats_normalize(rewards, update_normalization)
            elif self.reward_normalization == PredictedRewardNormalization.EXTREME_TRAINING_STATES:
                rewards = self.extreme_state_normalize(rewards, obses, update_normalization)
            elif self.reward_normalization == PredictedRewardNormalization.MANUAL:
                rewards = self.manual_normalize(rewards)
            elif self.reward_normalization == PredictedRewardNormalization.NORM_TRAINING_STATES:
                pass

        self.reward_call_n += 1

        assert_equal(rewards.shape, (n_steps,))
        return rewards

    def train(self, prefs_train: PrefDB, prefs_val: PrefDB, val_interval, verbose=True):
        """
        Train all ensemble members for one epoch.
        """

        if verbose:
            print("Training/testing with %d/%d preferences" % (len(prefs_train), len(prefs_val)))

        train_losses = []
        val_losses = []
        for batch_n, batch in enumerate(batch_iter(prefs_train.prefs, batch_size=32, shuffle=True)):
            train_losses.append(self.train_step(batch, prefs_train))
            self.n_steps += 1
            if self.n_steps % val_interval == 0 and len(prefs_val) != 0:
                val_losses.append(self.val_step(prefs_val))
            if self.n_steps % 100 == 0:
                rate = self.step_rate.measure(self.n_steps)
                self.logger.logkv('reward_predictor/training_steps_per_second', rate)
            if self.n_steps % 10 == 0:
                throttler.throttle_sleep(throttler.EventType.REWARD_PREDICTOR_10_STEPS)
                throttler.mark_event(throttler.EventType.REWARD_PREDICTOR_10_STEPS)

        if val_losses:
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            ratio = val_loss / (train_loss + 1e-8)
            self.logger.logkv('reward_predictor/test_train_loss_ratio', ratio)
            if ratio > 1.5:
                self.l2_reg_coef *= 1.5
            elif ratio < 1.1:
                self.l2_reg_coef = max(self.l2_reg_coef / 1.1, MIN_L2_REG_COEF)
            self.logger.logkv('reward_predictor/reg_coef', self.l2_reg_coef)

        self.n_epochs += 1
        self.log_n_epochs()

        if verbose:
            print("Done training DRLHP!")

    def log_n_epochs(self):
        self.logger.logkv('reward_predictor/n_epochs', self.n_epochs)

    def train_step(self, batch, prefs_train):
        s1s = [prefs_train.segments[k1] for k1, k2, pref, in batch]
        s2s = [prefs_train.segments[k2] for k1, k2, pref, in batch]
        prefs = [pref for k1, k2, pref, in batch]

        feed_dict = self.get_feed_dict()
        for rp in self.rps:
            feed_dict[rp.s1] = s1s
            feed_dict[rp.s2] = s2s
            feed_dict[rp.pref] = prefs
            feed_dict[rp.training] = True

        if self.reward_normalization == PredictedRewardNormalization.NORM_RANDOM_STATES:
            self.add_random_states_to_feed_dict(feed_dict)

        # Why do we only check the loss from the first reward predictor?
        # As a quick hack to get adaptive L2 regularization working quickly,
        # assuming we're only using one reward predictor.
        ops = [self.rps[0].prediction_loss, self.rps[0].rs_norm_loss,
               self.summaries, [rp.train for rp in self.rps]]
        with LogMilliseconds('instrumentation/reward_predictor_train_step_ms', self.logger, log_every=1000):
            loss, norm_loss, summaries, _ = self.sess.run(ops, feed_dict)
        if self.n_steps % self.log_interval == 0:
            self.train_writer.add_summary(summaries, self.n_steps)
            self.logger.logkv('reward_predictor/rs_norm_loss', norm_loss)
        return loss

    def val_step(self, prefs_val):
        val_batch_size = 32
        if len(prefs_val.prefs) <= val_batch_size:
            batch = prefs_val.prefs
        else:
            idxs = np.random.choice(len(prefs_val.prefs), val_batch_size, replace=False)
            batch = [prefs_val.prefs[i] for i in idxs]
        s1s = [prefs_val.segments[k1] for k1, k2, pref, in batch]
        s2s = [prefs_val.segments[k2] for k1, k2, pref, in batch]
        prefs = [pref for k1, k2, pref, in batch]
        feed_dict = self.get_feed_dict()
        for rp in self.rps:
            feed_dict[rp.s1] = s1s
            feed_dict[rp.s2] = s2s
            feed_dict[rp.pref] = prefs
            feed_dict[rp.training] = False

        if self.reward_normalization == PredictedRewardNormalization.NORM_RANDOM_STATES:
            self.add_random_states_to_feed_dict(feed_dict)

        loss, summaries = self.sess.run([self.rps[0].prediction_loss, self.summaries], feed_dict)
        if self.n_steps % self.log_interval == 0:
            self.test_writer.add_summary(summaries, self.n_steps)
        return loss

    def add_random_states_to_feed_dict(self, feed_dict):
        n_random_states = 16
        assert len(self.obs_space.shape) == 1
        random_states = np.random.uniform(self.obs_space.low, self.obs_space.high,
                                          size=[n_random_states, self.obs_space.shape[0]])
        for rp in self.rps:
            feed_dict[rp.random_states] = random_states

    def reset_normalisation(self):
        self.r_norm_limited = LimitedRunningStat()
        self.r_norm = RunningStat(shape=1)

    def get_feed_dict(self):
        feed_dict = {}
        for rp in self.rps:
            feed_dict[rp.l2_reg_coef] = self.l2_reg_coef
        return feed_dict


class RewardPredictorNetwork:
    """
    Predict the reward that a human would assign to each frame of
    the input trajectory, trained using the human's preferences between
    pairs of trajectories.

    Network inputs:
    - s1/s2     Trajectory pairs
    - pref      Preferences between each pair of trajectories
    Network outputs:
    - r1/r2     Reward predicted for each frame
    - rs1/rs2   Reward summed over all frames for each trajectory
    - pred      Predicted preference
    """

    def __init__(self, core_network, network_args, obs_shape, lr, reward_normalization: PredictedRewardNormalization,
                 normalization_loss_coef):
        training = tf.placeholder(tf.bool)
        # Each element of the batch is one trajectory segment.
        # (Dimensions are n segments x n frames per segment x ...)
        s1 = tf.placeholder(tf.float32, shape=(None, None) + obs_shape, name='s1')
        s2 = tf.placeholder(tf.float32, shape=(None, None) + obs_shape, name='s2')
        random_states = tf.placeholder(tf.float32, shape=(None,) + obs_shape, name='random_states')
        # For each trajectory segment, there is one human judgement.
        pref = tf.placeholder(tf.float32, shape=(None, 2), name='pref')

        # Concatenate trajectory segments so that the first dimension is just
        # frames
        # (necessary because of conv layer's requirements on input shape)
        s1_unrolled = tf.reshape(s1, (-1,) + obs_shape)
        s2_unrolled = tf.reshape(s2, (-1,) + obs_shape)

        l2_reg_coef = tf.placeholder(tf.float32)
        l2_reg = tf.contrib.layers.l2_regularizer(scale=l2_reg_coef)
        # Predict rewards for each frame in the unrolled batch
        _r1 = core_network(s=s1_unrolled, reuse=False, training=training, regularizer=l2_reg,
                           **network_args)
        _r2 = core_network(s=s2_unrolled, reuse=True, training=training, regularizer=l2_reg,
                           **network_args)
        r_random_states = core_network(s=random_states, reuse=True, training=False, regularizer=l2_reg,
                                       **network_args)

        # Shape should be 'unrolled batch size'
        # where 'unrolled batch size' is 'batch size' x 'n frames per segment'
        c1 = tf.assert_rank(_r1, 1)
        c2 = tf.assert_rank(_r2, 1)
        with tf.control_dependencies([c1, c2]):
            # Re-roll to 'batch size' x 'n frames per segment'
            __r1 = tf.reshape(_r1, tf.shape(s1)[0:2])
            __r2 = tf.reshape(_r2, tf.shape(s2)[0:2])
        # Shape should be 'batch size' x 'n frames per segment'
        c1 = tf.assert_rank(__r1, 2)
        c2 = tf.assert_rank(__r2, 2)
        with tf.control_dependencies([c1, c2]):
            r1 = __r1
            r2 = __r2

        # Sum rewards over all frames in each segment
        _rs1 = tf.reduce_sum(r1, axis=1)
        _rs2 = tf.reduce_sum(r2, axis=1)
        # Shape should be 'batch size'
        c1 = tf.assert_rank(_rs1, 1)
        c2 = tf.assert_rank(_rs2, 1)
        with tf.control_dependencies([c1, c2]):
            rs1 = _rs1
            rs2 = _rs2

        # Predict preferences for each segment
        _rs = tf.stack([rs1, rs2], axis=1)
        # Shape should be 'batch size' x 2
        c1 = tf.assert_rank(_rs, 2)
        with tf.control_dependencies([c1]):
            rs = _rs

        _pred = tf.nn.softmax(rs)
        # Shape should be 'batch_size' x 2
        c1 = tf.assert_rank(_pred, 2)
        with tf.control_dependencies([c1]):
            pred = _pred

        preds_correct = tf.equal(tf.argmax(pref, 1), tf.argmax(pred, 1))
        accuracy = tf.reduce_mean(tf.cast(preds_correct, tf.float32))

        _loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pref,
                                                           logits=rs)
        # Shape should be 'batch size'
        c1 = tf.assert_rank(_loss, 1)
        with tf.control_dependencies([c1]):
            loss = tf.reduce_sum(_loss)

        self.prediction_loss = loss

        l2_reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # l2_reg_losses is a list of L2 norms - one for each weight layer
        # (where each L2 norm is just a scalar - so this is a list of scalars)
        # Why do we use add_n rather than reduce_sum?
        # reduce_sum is for when you have e.g. a matrix and you want to sum over one row.
        # If you want to sum over elements of a list, you use add_n.
        l2_reg_loss = tf.add_n(l2_reg_losses)
        loss += l2_reg_loss

        if reward_normalization == PredictedRewardNormalization.NORM_TRAINING_STATES:
            reward_norm_loss = tf.norm(r1, ord=1) + tf.norm(r2, ord=1)
        elif reward_normalization == PredictedRewardNormalization.NORM_RANDOM_STATES:
            reward_norm_loss = tf.norm(r_random_states, ord=1)
        else:
            reward_norm_loss = tf.constant(0.0)
        loss += normalization_loss_coef * reward_norm_loss

        if core_network == net_cnn:
            batchnorm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            optimizer_dependencies = batchnorm_update_ops
        else:
            optimizer_dependencies = []

        with tf.control_dependencies(optimizer_dependencies):
            train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        # Inputs
        self.training = training
        self.s1 = s1
        self.s2 = s2
        self.random_states = random_states
        self.pref = pref
        self.l2_reg_coef = l2_reg_coef

        # Outputs
        self.r1 = r1
        self.r2 = r2
        self.rs1 = rs1
        self.rs2 = rs2
        self.pred = pred

        self.accuracy = accuracy
        self.loss = loss
        self.train = train
        self.l2_reg_loss = l2_reg_loss

        self.rs_norm_loss = reward_norm_loss
