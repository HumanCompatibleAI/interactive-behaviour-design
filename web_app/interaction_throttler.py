import easy_tf_log

import global_variables
from web_app import web_globals
from web_app.utils import get_n_rl_steps

logger = easy_tf_log.Logger(web_globals._log_dir)
n_rl_steps_at_last_interaction = None


def mark_interaction():
    n_rl_steps = get_n_rl_steps()
    # If we haven't started training yet (i.e. we're still pretraining)
    if n_rl_steps is None:
        return
    logger.logkv('interaction_limit/n_rl_steps_at_last_interaction', n_rl_steps)


def throttle(interaction_type):
    # 0 => don't throttle
    if global_variables.n_rl_steps_per_interaction == 0:
        return
    if n_rl_steps_at_last_interaction is None:
        return
    n_rl_steps = get_n_rl_steps()
    if n_rl_steps is None:
        return

    n_rl_steps_since_last_interaction = n_rl_steps - n_rl_steps_at_last_interaction
    logger.logkv('interaction_limit/n_rl_steps', n_rl_steps)
    logger.logkv('interaction_limit/n_rl_steps_since_last_pref', n_rl_steps_since_last_interaction)
    if n_rl_steps_since_last_interaction < global_variables.n_rl_steps_per_interaction:
        return f'No {interaction_type} available'
