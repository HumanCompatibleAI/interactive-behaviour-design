import multiprocessing
from typing import Dict

from checkpointer import Checkpointer
from classifier_buffer import ClassifierDataBuffer
from classifier_collection import ClassifierCollection
from policies.policy_collection import PolicyCollection
from policies.td3 import LockedReplayBuffer
from policy_rollouter import PolicyRollouter
from rollouts import RolloutsByHash, RolloutHash, CompressedRollout
from drlhp.pref_db import PrefDBTestTrain
from wrappers.util_wrappers import VecRewardSwitcherWrapper, ResetStateCache

_classifiers = None  # type: ClassifierCollection
_policies = None  # type: PolicyCollection
_set_reward_source_fn = None  # type: function
global_experience_buffer = None  # type: ClassifierDataBuffer
save_dir = None  # type: str
experience_dir = None  # type: str
_cur_label = None
_pref_db = None  # type: PrefDBTestTrain
_reward_switcher_wrapper = None  # type: VecRewardSwitcherWrapper
_demo_env = None
_policy_rollouter = None  # type: PolicyRollouter
_log_dir = None  # type: str
_demonstration_rollouts = None  # type: RolloutsByHash
_reset_mode_value = None  # type: multiprocessing.Value
_reset_state_cache = None  # type: ResetStateCache
_max_episode_steps_value = None  # type: multiprocessing.Value
_cur_rollouts = {}  # type: Dict[RolloutHash, CompressedRollout]
_save_state_from_proportion_through_episode_value = None  # type: multiprocessing.Value
_demonstrations_reset_mode_value = None  # type: multiprocessing.Value
_run_drlhp_training = None  # type: multiprocessing.Value
_demonstration_rollouts_dir = None  # type: str
_segments_dir = None  # type: str
FPS = 30.0
_checkpointer = None  # type: Checkpointer
_max_demonstration_length = None # type: int
_demonstrations_replay_buffer = None  # type: LockedReplayBuffer
