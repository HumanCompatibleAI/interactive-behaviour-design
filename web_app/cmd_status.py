import contextlib
import glob
import io
import json
import os
import sys

from flask import request, render_template, Blueprint

from policies.base_policy import PolicyTrainMode
from reward_switcher import RewardSource
from web_app.utils import check_bc_losses
from web_app.view_trajectories import get_trajectories
from web_app.web_globals import _classifiers, _cur_label, _demonstration_rollouts, _log_dir, \
    _max_episode_steps_value, _policies, _pref_db, _reset_mode_value, _reset_state_cache, \
    _reward_selector, \
    global_experience_buffer, _save_state_from_proportion_through_episode_value, \
    _demonstrations_reset_mode_value, \
    _run_drlhp_training, _checkpointer, _policy_rollouter
from wrappers.util_wrappers import ResetMode, QueueEndpoint

cmd_status_app = Blueprint('cmd_status', __name__)

train_thread = None


@cmd_status_app.route('/run_cmd', methods=['GET'])
def run_cmd():
    global train_thread
    cmd = request.args['cmd']
    print("Got command '{}'".format(cmd))

    if cmd == 'train':
        n_epochs = int(request.args['n_epochs'])
        if _reward_selector.cur_classifier_name is None:
            return "Error: no classifier selected"
        for _ in range(n_epochs):
            _classifiers.train(_reward_selector.cur_classifier_name)
            _classifiers.test(_reward_selector.cur_classifier_name)
    elif cmd == 'reset_classifier':
        _classifiers.reset(_cur_label)
    elif cmd == 'reset_experience':
        global_experience_buffer.reset_labels()
    elif cmd == 'set_reward_source':
        src_str = request.args['src']
        if src_str == 'env':
            src = RewardSource.ENV
        elif src_str == 'classifier':
            src = RewardSource.CLASSIFIER
            if _reward_selector.cur_classifier_name is None:
                return "cur_classifier is None"
        elif src_str == 'drlhp':
            src = RewardSource.DRLHP
        elif src_str == 'none':
            src = RewardSource.NONE
        else:
            return "Unknown reward source '{}'".format(src_str)
        _reward_selector.set_reward_source(src)
    elif cmd == 'add_classifier':
        label_name = request.args['name']
        _classifiers.add_classifier(label_name)
    elif cmd == 'use_classifier':
        label_name = request.args['name']
        if label_name not in _reward_selector.classifiers.classifiers:
            return "Unknown classifier: '{}'".format(label_name)
        _reward_selector.cur_classifier_name = label_name
    elif cmd == 'add_policy':
        policy_name = request.args['name']
        if policy_name in _policies.policies:
            msg = "Policy '{}' already exists".format(policy_name)
            print(msg)
            return msg
        _policies.add_policy(policy_name)
        # In case we want to use the added policy for demonstrations straight away
        _checkpointer.checkpoint()
    elif cmd == 'rm_policy':
        policy_name = request.args['name']
        if policy_name not in _policies.policies:
            return "Policy '{}' doesn't exist".format(policy_name)
        if _policies.cur_policy == policy_name:
            _policies.cur_policy = None
        del _policies.policies[policy_name]
    elif cmd == 'use_policy':
        policy_name = request.args['name']
        if policy_name == 'none':
            policy_name = None
        elif policy_name not in _policies.policies:
            return "Policy '{}' not defined".format(policy_name)
        _policies.set_active_policy(policy_name)
    elif cmd == 'start_drlhp_training':
        _run_drlhp_training.value = 1
    elif cmd == 'stop_drlhp_training':
        _run_drlhp_training.value = 0
    elif cmd == 'training_mode':
        mode_str = request.args['mode']
        if _policies.cur_policy is None:
            return "Current policy not defined"
        if mode_str == 'reward_only':
            mode = PolicyTrainMode.R_ONLY
        elif mode_str == 'reward_plus_bc':
            mode = PolicyTrainMode.R_PLUS_BC
        elif mode_str == 'reward_plus_sqil':
            mode = PolicyTrainMode.R_PLUS_SQIL
        elif mode_str == 'bc_only':
            mode = PolicyTrainMode.BC_ONLY
        elif mode_str == 'sqil_only':
            mode = PolicyTrainMode.SQIL_ONLY
        elif mode_str == 'no_training':
            mode = PolicyTrainMode.NO_TRAINING
        else:
            return "Invalid mode '{}'".format(mode_str)
        _policies[_policies.cur_policy].train_mode = mode
    elif cmd == 'set_bc_coef':
        coef = float(request.args['coef'])
        _policies.set_bc_coef(coef)
    elif cmd == 'set_ent_coef':
        coef = float(request.args['coef'])
        _policies.set_ent_coef(coef)
    elif cmd == 'load_rp_ckpt':
        ckpt_n = request.args['n']

        search = os.path.join(_log_dir, 'checkpoints', 'drlhp*-{}.meta'.format(ckpt_n))
        possible_ckpts = glob.glob(search)
        if len(possible_ckpts) == 0:
            return "No checkpoint found"
        if len(possible_ckpts) > 1:
            return "ckpt_n not specific"
        ckpt_path = possible_ckpts[0].replace('.meta', '')

        _reward_selector.reward_predictor.load(ckpt_path)
    elif cmd == 'load_policy_ckpt':
        policy = request.args['policy']
        ckpt_n = request.args['n']

        if policy != _policies.cur_policy:
            return "Current policy is not '{}'".format(policy)

        search = os.path.join(_log_dir, 'checkpoints', 'policy-{}-*-{}.meta'.format(policy, ckpt_n))
        possible_ckpts = glob.glob(search)
        if len(possible_ckpts) == 0:
            return "No checkpoint found"
        if len(possible_ckpts) > 1:
            return "ckpt_n not specific"
        ckpt_path = possible_ckpts[0].replace('.meta', '')

        _policies[_policies.cur_policy].load_checkpoint(ckpt_path)
    elif cmd == 'reset_drlhp_normalisation':
        _reward_selector.reward_predictor.reset_normalisation()
    elif cmd == 'check_bc_losses':
        losses_dict = check_bc_losses()
        return str(losses_dict)
    elif cmd == 'clear_state_cache':
        _reset_state_cache.clear()
    elif cmd in ['set_training_reset_mode', 'set_demonstrations_reset_mode']:
        mode_str = request.args['mode']
        if mode_str == 'use_env_reset':
            mode = ResetMode.USE_ENV_RESET.value
        elif mode_str == 'from_state_cache':
            mode = ResetMode.FROM_STATE_CACHE.value
        else:
            msg = "Warning: reset mode '{}' not recognised".format(mode_str)
            print(msg)
            return msg
        if 'training' in cmd:
            _reset_mode_value.value = mode
        elif 'demonstrations' in cmd:
            _demonstrations_reset_mode_value.value = mode
    elif cmd == 'reset_demonstrations':
        _demonstration_rollouts.clear()
    elif cmd == 'reset_prefs':
        _pref_db.reset()
    elif cmd == 'add_reset_pool':
        name = request.args['name']
        try:
            max_len = int(request.args['max_len'])
        except:
            max_len = None
        _reset_state_cache.add_pool(name, max_len)
        return ""
    elif cmd == 'use_reset_pool':
        pool_name = request.args['name']
        if pool_name not in _reset_state_cache.pools:
            return "Reset state pool '{}' not found".format(pool_name)
        if 'from' in request.args:
            endpoint_str = request.args['from']
            set_dict = _reset_state_cache.receive_to_pool
        elif 'to' in request.args:
            endpoint_str = request.args['to']
            set_dict = _reset_state_cache.serve_from_pool
        else:
            return "Neither 'from' nor 'to' specified"
        if endpoint_str == 'training':
            endpoint = QueueEndpoint.TRAINING
        elif endpoint_str == 'demonstrations':
            endpoint = QueueEndpoint.DEMONSTRATIONS
        else:
            return "Endpoint must be 'training' or 'demonstrations'"
        set_dict[endpoint] = pool_name
    elif cmd == 'set_max_episode_steps':
        n = int(request.args['n'])
        _max_episode_steps_value.value = n
    elif cmd == 'set_save_proportion':
        p = float(request.args['p'])
        _save_state_from_proportion_through_episode_value.value = p
    elif cmd == 'set_bc_ent_coef':
        coef = float(request.args['coef'])
        _policies.set_bc_ent_coef(coef)
    elif cmd == 'train_bc':
        if _policies.cur_policy is None:
            return "cur_policy is none"
        n_epochs = int(request.args['n_epochs'])
        for _ in range(n_epochs):
            _policies[_policies.cur_policy].train_bc_epoch()
    elif cmd == 'set_drlhp_l2_loss_coef':
        coef = float(request.args['coef'])
        _reward_selector.reward_predictor.l2_loss_coef = coef
    elif cmd == 'eval':
        eval_str = request.args['str']
        orig_stdout = sys.stdout
        stdout_string = io.StringIO()
        sys.stdout = stdout_string
        try:
            exec(eval_str)
            return stdout_string.getvalue()
        except Exception as e:
            return (str(e))
        finally:
            sys.stdout = orig_stdout
    elif cmd == 'config_demos':
        noop_actions = request.args['noop_actions']
        if noop_actions == 'True':
            noop_actions = True
        elif noop_actions == 'False':
            noop_actions = False
        else:
            raise Exception(f"Invalid noop actions value '{noop_actions}")
        _policy_rollouter.noop_actions = noop_actions
    elif cmd == 'checkpoint':
        _checkpointer.checkpoint()
    else:
        return "Unknown command '{}'".format(cmd)

    return ""


@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = io.StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old


@cmd_status_app.route('/status', methods=['GET'])
def status():
    return render_template('status.html')


@cmd_status_app.route('/cmd_history', methods=['GET'])
def cmd_history():
    return render_template('cmd_history.html')


@cmd_status_app.route('/get_status', methods=['GET'])
def get_status():
    if _policies.cur_policy is None:
        training_mode = None
    else:
        training_mode = _policies[_policies.cur_policy].train_mode

    reset_state_counts = {pool_name: len(_reset_state_cache.pools[pool_name])
                          for pool_name in _reset_state_cache.pools}

    status_dict = {
        'Policies': ','.join(_policies.policies.keys()),
        'Current policy': _policies.cur_policy,
        'Classifiers': ','.join(_classifiers.classifiers.keys()),
        'Current classifier': _reward_selector.cur_classifier_name,
        'Reward source': str(_reward_selector.cur_reward_source),
        'Policy training mode': str(training_mode),
        'Training reset mode': str(ResetMode(_reset_mode_value.value)),
        'Demonstrations reset mode': str(ResetMode(_demonstrations_reset_mode_value.value)),
        'Max. episode steps': str(_max_episode_steps_value.value),
        'Save states from proportion through episode': str(
            _save_state_from_proportion_through_episode_value.value),
        'Number of demonstration rollouts': len(_demonstration_rollouts),
        'Reset state counts': str(reset_state_counts),
        'Reset states received to pool': str(_reset_state_cache.receive_to_pool),
        'Reset states sent from pool': str(_reset_state_cache.serve_from_pool),
        'No. prefs': str(len(_pref_db.train)),
        'No. demonstrated episodes': str(len(get_trajectories())),
        'No. demonstrations': len(_demonstration_rollouts),
    }

    return json.dumps(status_dict)
