import gzip
import time
import uuid

import hashlib
import os
import pickle
import tempfile
from collections import deque
from threading import Lock
from typing import List, Dict

import magic
import numpy as np

from utils import CompressedAttributes, EnvState


class Rollout():
    def __init__(self, final_env_state: EnvState,
                 obses: List, actions: List[int], rewards: List[float], frames: List,
                 vid_filename: str = None, generating_policy: str = None):
        self.final_env_state = final_env_state
        self.obses = obses
        self.actions = actions
        self.rewards = rewards
        self.frames = frames
        self.vid_filename = vid_filename
        self.generating_policy = generating_policy
        self.hash = RolloutHash(str(time.time()))

    def __hash__(self):
        return self.hash.__hash__()

    def __eq__(self, other):
        return self.hash == other.hash


class CompressedRollout(CompressedAttributes):
    def __init__(self, final_env_state: EnvState,
                 obses: List, actions: List[int], rewards: List[float], frames: List,
                 vid_filename: str = None, generating_policy: str = None, extra_info=None):
        CompressedAttributes.__init__(self)
        self.final_env_state = final_env_state
        self.obses = obses
        self.actions = actions
        self.rewards = rewards
        self.frames = frames
        self.vid_filename = vid_filename
        self.generating_policy = generating_policy
        self.uuid = uuid.uuid4()
        # NB this is not actually a hash. It used to be but that caused problems. Now it's more like an ID.
        self.hash = RolloutHash(str(self.uuid))
        self.extra_info = extra_info

    def __hash__(self):
        return self.hash.__hash__()

    def __eq__(self, other):
        return self.hash == other.hash


class RolloutHash:

    def __init__(self, t):
        self.hash = hashlib.md5(t.encode()).hexdigest()[:8]

    def __hash__(self):
        return hash(self.hash)

    def __eq__(self, other):
        return self.hash == other.hash

    def __str__(self):
        return self.hash


class RolloutsByHash:
    """
    Thread-safe dictionary with type hinting and length limiting.
    """

    def __init__(self, maxlen=None):
        self.dict = dict()  # type: Dict[RolloutHash, CompressedRollout]
        self.losses = dict()
        # Prevent one thread for iterating while another thread is modifying
        self.lock = Lock()
        self.keys_in_insertion_order = deque()
        self.maxlen = maxlen

    def __getitem__(self, key: RolloutHash):
        rollout: CompressedRollout = self.dict[key]
        return rollout

    def __setitem__(self, key: RolloutHash, value: CompressedRollout):
        self.lock.acquire()

        if key in self.dict:
            print(f"Warning: rollout '{str(key.hash)}' is already in dictionary")
        self.dict[key] = value

        if key in self.keys_in_insertion_order:
            self.keys_in_insertion_order.remove(key)
        self.keys_in_insertion_order.append(key)

        if self.maxlen is not None and len(self) > self.maxlen:
            key_to_delete = self.keys_in_insertion_order.popleft()
            del self.dict[key_to_delete]

        self.lock.release()

    def __len__(self):
        return len(self.dict)

    def clear(self):
        self.dict = dict()

    def keys(self):
        self.lock.acquire()
        keys = list(self.dict.keys())
        self.lock.release()
        return keys

    def values(self):
        self.lock.acquire()
        values = list(self.dict.values())
        self.lock.release()
        return values

    def save(self, save_path):
        save_dir = os.path.dirname(save_path)
        fd, temp_file = tempfile.mkstemp(dir=save_dir)
        self.lock.acquire()
        with open(temp_file, 'wb') as f:
            pickle.dump(self.dict, f)
        self.lock.release()
        os.close(fd)
        os.rename(temp_file, save_path)

    def load(self, path):
        if 'gzip' in magic.from_file(path):
            open_f = gzip.open
        else:
            open_f = open
        with open_f(path, 'rb') as f:
            self.dict = pickle.load(f)
            print("Loaded {} demonstration rollouts".format(len(self.dict)))
