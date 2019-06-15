from datetime import datetime
from functools import wraps, update_wrapper

import easy_tf_log
import numpy as np
from flask import make_response

import web_app.web_globals as web_globals
from rollouts import CompressedRollout

logger = None


def init_web_logger():
    global logger
    logger = easy_tf_log.Logger()
    logger.set_log_dir(web_globals.experience_dir)


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = ('no-store, no-cache, '
                                             'must-revalidate, post-check=0, '
                                             'pre-check=0, max-age=0')
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response

    return update_wrapper(no_cache, view)


def check_bc_losses():
    if web_globals._policies.cur_policy is None:
        raise Exception("cur_policy is none")
    pol = web_globals._policies[web_globals._policies.cur_policy]
    losses = dict()
    for rollout_frames_hash in web_globals._demonstration_rollouts.keys():
        rollout = web_globals._demonstration_rollouts[rollout_frames_hash]
        bc_loss = pol.model.check_bc_loss(rollout.frames, rollout.actions)
        losses[rollout_frames_hash.hash] = bc_loss
    return losses


def add_pref(rollout1: CompressedRollout, rollout2: CompressedRollout, pref):
    if np.allclose(rollout1.obses, rollout2.obses):
        print(f"Dropping preference for {rollout1.hash} and {rollout2.hash} because identical")
        return

    msg = f"Adding preference {pref} for {rollout1.hash} vs {rollout2.hash}"
    if rollout1.generating_policy is not None:
        msg += f" (policies {rollout1.generating_policy} vs. {rollout2.generating_policy})"
    print(msg)
    web_globals._pref_db.append(rollout1.obses, rollout2.obses, pref)
    add_pref.added_prefs += 1
    logger.logkv('pref_db/n_prefs', len(web_globals._pref_db.train))
    logger.logkv('pref_db/added_prefs', add_pref.added_prefs)


add_pref.added_prefs = 0
