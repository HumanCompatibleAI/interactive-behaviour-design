import copy
import os
import pickle
import queue
import random
import sys
import threading
import time
from threading import Lock, Thread

import easy_tf_log
import numpy as np
from gym.utils import atomic_write

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from global_constants import MAX_PREFS


class Segment:
    """
    A short recording of agent's behaviour in the environment,
    consisting of a number of video frames and the rewards it received
    during those frames.
    """

    def __init__(self):
        self.frames = []
        self.rewards = []
        self.hash = None

    def append(self, frame, reward):
        self.frames.append(frame)
        self.rewards.append(reward)

    def finalise(self, seg_id=None):
        if seg_id is not None:
            self.hash = seg_id
        else:
            # This looks expensive, but don't worry -
            # it only takes about 0.5 ms.
            self.hash = hash(np.array(self.frames).tostring())

    def __len__(self):
        return len(self.frames)


class PrefDB:
    """
    A circular database of preferences about pairs of segments.

    For each preference, we store the preference itself
    (mu in the paper) and the two segments the preference refers to.
    Segments are stored with deduplication - so that if multiple
    preferences refer to the same segment, the segment is only stored once.
    """

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.lock = threading.Lock()
        self.segments = self.seg_refs = self.prefs = None
        self.reset()

    def __getstate__(self):
        with self.lock:
            return (self.maxlen, self.segments, self.seg_refs, self.prefs)

    def __setstate__(self, state):
        self.maxlen, self.segments, self.seg_refs, self.prefs = state
        self.lock = threading.Lock()

    def copy(self):
        p = PrefDB(self.maxlen)
        with self.lock:
            p.segments = copy.deepcopy(self.segments)
            p.seg_refs = copy.deepcopy(self.seg_refs)
            p.prefs = copy.deepcopy(self.prefs)
        return p

    def reset(self):
        with self.lock:
            self.segments = dict()
            self.seg_refs = {}
            self.prefs = []

    def append(self, s1, s2, pref):
        with self.lock:
            k1 = hash(np.array(s1).tostring())
            k2 = hash(np.array(s2).tostring())

            for k, s in zip([k1, k2], [s1, s2]):
                if k not in self.segments.keys():
                    self.segments[k] = s
                    self.seg_refs[k] = 1
                else:
                    self.seg_refs[k] += 1

            tup = (k1, k2, pref)
            self.prefs.append(tup)

        if len(self.prefs) > self.maxlen:
            self.del_first()

    def del_first(self):
        self.del_pref(0)

    def del_pref(self, n):
        if n >= len(self.prefs):
            raise IndexError("Preference {} doesn't exist".format(n))
        with self.lock:
            k1, k2, _ = self.prefs[n]
            for k in [k1, k2]:
                if self.seg_refs[k] == 1:
                    del self.segments[k]
                    del self.seg_refs[k]
                else:
                    self.seg_refs[k] -= 1
            del self.prefs[n]

    def __len__(self):
        return len(self.prefs)


class PrefBuffer:
    """
    A helper class to manage asynchronous receiving of preferences on a
    background thread.
    """

    def __init__(self, db_train, db_val):
        self.train_db = db_train
        self.val_db = db_val
        self.lock = Lock()
        self.stop_recv = False

    def start_recv_thread(self, pref_pipe):
        self.stop_recv = False
        Thread(target=self.recv_prefs, args=(pref_pipe,)).start()

    def stop_recv_thread(self):
        self.stop_recv = True

    def recv_prefs(self, pref_pipe):
        n_recvd = 0
        while not self.stop_recv:
            try:
                s1, s2, pref = pref_pipe.get(block=True, timeout=1)
            except queue.Empty:
                continue
            n_recvd += 1

            val_fraction = self.val_db.maxlen / (self.val_db.maxlen +
                                                 self.train_db.maxlen)

            self.lock.acquire(blocking=True)
            if np.random.rand() < val_fraction:
                self.val_db.append(s1, s2, pref)
                easy_tf_log.tflog('val_db_len', len(self.val_db))
            else:
                self.train_db.append(s1, s2, pref)
                easy_tf_log.tflog('train_db_len', len(self.train_db))
            self.lock.release()

            easy_tf_log.tflog('n_prefs_recvd', n_recvd)

    def train_db_len(self):
        return len(self.train_db)

    def val_db_len(self):
        return len(self.val_db)

    def get_dbs(self):
        self.lock.acquire(blocking=True)
        train_copy = copy.deepcopy(self.train_db)
        val_copy = copy.deepcopy(self.val_db)
        self.lock.release()
        return train_copy, val_copy

    def wait_until_len(self, min_len):
        while True:
            self.lock.acquire()
            train_len = len(self.train_db)
            val_len = len(self.val_db)
            self.lock.release()
            if train_len >= min_len and val_len != 0:
                break
            print("Waiting for preferences; {} so far".format(train_len))
            time.sleep(5.0)


class PrefDBTestTrain:
    def __init__(self):
        self.reset()

    def load(self, path, verbose=True):
        with open(path, 'rb') as f:
            self.test, self.train = pickle.load(f)
            if verbose:
                print(f"Loaded {len(self.test)}/{len(self.train)} test/train prefs")

    def save(self, path):
        # Important to write atomically because this is loaded by the reward predictor training process
        with atomic_write.atomic_write(path, binary=True) as f:
            pickle.dump((self.test, self.train), f)

    def append(self, s1, s2, pref):
        if random.random() < 0.2:
            self.test.append(s1, s2, pref)
        else:
            self.train.append(s1, s2, pref)

    def __len__(self):
        return len(self.test) + len(self.train)

    def reset(self):
        self.test = PrefDB(maxlen=MAX_PREFS)
        self.train = PrefDB(maxlen=MAX_PREFS)
