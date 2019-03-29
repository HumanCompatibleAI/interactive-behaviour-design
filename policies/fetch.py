import os
import re
from enum import Enum

from policies.td3 import TD3Policy


class FetchTD3Policy(TD3Policy):
    def __init__(self, name, env_id, obs_space, ac_space, n_envs, seed=0, fetch_action=None):
        super().__init__(name, env_id, obs_space, ac_space, n_envs, seed)
        self.manual_action = fetch_action

    def step(self, obs, **step_kwargs):
        if self.manual_action is None:
            return super().step(obs, **step_kwargs)

        elif self.manual_action == FetchAction.BACKWARDOPEN:
            return [0.2, 0, 0, +1.0]
        elif self.manual_action == FetchAction.FORWARDOPEN:
            return [-0.2, 0, 0, +1.0]
        elif self.manual_action == FetchAction.LEFTOPEN:
            return [0, 0.2, 0, +1.0]
        elif self.manual_action == FetchAction.RIGHTOPEN:
            return [0, -0.2, 0, +1.0]
        elif self.manual_action == FetchAction.UPOPEN:
            return [0, 0, 0.2, +1.0]
        elif self.manual_action == FetchAction.DOWNOPEN:
            return [0, 0, -0.2, +1.0]

        elif self.manual_action == FetchAction.BACKWARDCLOSE:
            return [0.2, 0, 0, -1.0]
        elif self.manual_action == FetchAction.FORWARDCLOSE:
            return [-0.2, 0, 0, -1.0]
        elif self.manual_action == FetchAction.LEFTCLOSE:
            return [0, 0.2, 0, -1.0]
        elif self.manual_action == FetchAction.RIGHTCLOSE:
            return [0, -0.2, 0, -1.0]
        elif self.manual_action == FetchAction.UPCLOSE:
            return [0, 0, 0.2, -1.0]
        elif self.manual_action == FetchAction.DOWNCLOSE:
            return [0, 0, -0.2, -1.0]

    def load_checkpoint(self, path):
        if 'FetchAction' in path:
            action = re.search(r'FetchAction\.([A-Z]*)', path).group(1)
            self.manual_action = getattr(FetchAction, action)
            print("Restored policy checkpoint from '{}'".format(path))
        else:
            self.manual_action = None
            super().load_checkpoint(path)

    def save_checkpoint(self, path):
        if self.manual_action is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # find_latest_checkpoint looks for *.meta files
            open(path + '.meta', 'w').close()
        else:
            super().save_checkpoint(path)


class FetchAction(Enum):
    UPOPEN = 0
    DOWNOPEN = 1
    LEFTOPEN = 2
    RIGHTOPEN = 3
    FORWARDOPEN = 4
    BACKWARDOPEN = 5

    UPCLOSE = 6
    DOWNCLOSE = 7
    LEFTCLOSE = 8
    RIGHTCLOSE = 9
    FORWARDCLOSE = 10
    BACKWARDCLOSE = 11
