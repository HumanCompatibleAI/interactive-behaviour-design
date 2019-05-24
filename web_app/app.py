import logging
import os
import os.path as osp

import web_app.web_globals as web_globals
from flask import Flask


def run_web_app(classifiers, policies, reward_selector, experience_buffer, log_dir, port,
                pref_db, demo_env, policy_rollouter, demonstration_rollouts,
                reset_mode_value, reset_state_cache, max_episode_steps_value,
                save_state_from_proportion_through_episode_value, demonstrations_reset_mode_value,
                run_drlhp_training, rollout_vids_dir, segments_dir, checkpointer,
                max_demonstration_length):
    web_globals._classifiers = classifiers
    web_globals._policies = policies
    web_globals._pref_db = pref_db
    web_globals._reward_selector = reward_selector
    web_globals.global_experience_buffer = experience_buffer
    web_globals.experience_dir = osp.join(log_dir, 'experience')
    web_globals.save_dir = osp.join(log_dir, 'web')
    os.makedirs(web_globals.save_dir)
    web_globals._log_dir = log_dir
    web_globals._demo_env = demo_env
    web_globals._policy_rollouter = policy_rollouter
    web_globals._demonstration_rollouts = demonstration_rollouts
    web_globals._reset_mode_value = reset_mode_value
    web_globals._reset_state_cache = reset_state_cache
    web_globals._max_episode_steps_value = max_episode_steps_value
    web_globals._save_state_from_proportion_through_episode_value = save_state_from_proportion_through_episode_value
    web_globals._demonstrations_reset_mode_value = demonstrations_reset_mode_value
    web_globals._run_drlhp_training = run_drlhp_training
    web_globals._demonstration_rollouts_dir = rollout_vids_dir
    web_globals._segments_dir = segments_dir
    web_globals._checkpointer = checkpointer
    web_globals._max_demonstration_length = max_demonstration_length

    from web_app.cmd_status import cmd_status_app
    from web_app.demonstrations import demonstrations_app
    from web_app.labelling import labelling_app
    from web_app.comparisons import comparisons_app
    from web_app.view_trajectories import view_trajectories_app

    logging.basicConfig(filename=os.path.join(log_dir, 'flask.log'), level=logging.INFO)

    app = Flask(__name__)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.register_blueprint(comparisons_app)
    app.register_blueprint(cmd_status_app)
    app.register_blueprint(labelling_app)
    app.register_blueprint(demonstrations_app)
    app.register_blueprint(view_trajectories_app)

    app.run(port=port)
