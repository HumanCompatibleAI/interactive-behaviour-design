import datetime
import os
import threading
import time
from threading import Thread

import global_constants
from classifier_collection import ClassifierCollection
from policies.policy_collection import PolicyCollection
from drlhp.pref_db import PrefDBTestTrain


# Reward predictor checkpointing is handled by the reward predictor training loop in a separate process

class Checkpointer:
    def __init__(self, ckpt_dir: str,
                 policy_collection: PolicyCollection,
                 classifier_collection: ClassifierCollection,
                 pref_dbs: PrefDBTestTrain,
                 pref_dbs_ckpt_name):
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir)
        self.policy_collection = policy_collection
        self.classifier_collection = classifier_collection
        self.pref_dbs = pref_dbs
        self.pref_dbs_ckpt_name = pref_dbs_ckpt_name
        self.lock = threading.Lock()

        def checkpoint_loop():
            while True:
                time.sleep(global_constants.CKPT_EVERY_N_SECONDS)
                self.checkpoint()
        Thread(target=checkpoint_loop).start()

    def checkpoint(self):
        # Why lock? Because the checkpoint_loop thread will run this regularly,
        # but we also might want to call this method manually
        self.lock.acquire()

        now = str(datetime.datetime.now())

        for policy_name in dict(self.policy_collection.policies):  # make a copy in case changed while iterating
            policy_ckpt_path = os.path.join(self.ckpt_dir,
                                            'policy-{}-{}.ckpt'.format(policy_name, now))
            policy = self.policy_collection[policy_name]
            policy.save_checkpoint(policy_ckpt_path)

        if len(self.classifier_collection.classifiers) > 0:  # exist classifiers to save
            classifier_ckpt_path = os.path.join(self.ckpt_dir, 'classifiers-{}.ckpt'.format(now))
            self.classifier_collection.save_checkpoint(classifier_ckpt_path)

            # also save the classifier names for restoration
            classifier_names_path = os.path.join(self.ckpt_dir, 'classifier_names.txt'.format(now))
            f = open(classifier_names_path, "w")  # clear file (overwrite)
            for classifier_name in self.classifier_collection.classifiers:
                f = open(classifier_names_path, "a")  # append to file
                f.write(classifier_name + "\n")
            f.close()

        classifier_ckpt_path = os.path.join(self.ckpt_dir, 'classifiers-{}.ckpt'.format(now))
        self.classifier_collection.save_checkpoint(classifier_ckpt_path)
        # also save the classifier names for restoration
        classifier_names_path = os.path.join(self.ckpt_dir, 'classifier_names.txt'.format(now))
        f = open(classifier_names_path, "w")  # clear file (overwrite)
        for classifier_name in self.classifier_collection.classifiers:
            f = open(classifier_names_path, "a")  # append to file
            f.write(classifier_name + "\n")
        f.close()

        self.pref_dbs.save(os.path.join(self.ckpt_dir, self.pref_dbs_ckpt_name))

        self.lock.release()
