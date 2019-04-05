import multiprocessing
from typing import Dict

import web_app
from policies.base_policy import Policy
from policies.td3 import TD3Policy
from rollouts import RolloutsByHash


class PolicyCollection:
    policies: Dict[str, Policy]

    def __init__(self, make_policy_fn, log_dir, demonstrations: RolloutsByHash, seed,
                 eval_policy_name_queue: multiprocessing.Queue):
        self.policies = {}
        self.cur_policy = None
        self.make_policy = make_policy_fn
        self.env = None
        self.make_eval_env_fn = None
        self.ckpt_dir = None
        self.log_dir = log_dir
        self.demonstrations = demonstrations
        self.seed = seed
        self.eval_policy_name_queue = eval_policy_name_queue

    def add_policy(self, name, policy_kwargs=None):
        policy_kwargs = policy_kwargs or {}
        policy_kwargs.update({'seed': self.seed})
        self.policies[name] = self.make_policy(name, **policy_kwargs)
        print(f"Added policy '{name}'")

    def set_active_policy(self, name):
        for policy in self.policies.values():
            policy.stop_training()
        if name is not None:
            policy = self.policies[name]
            policy.init_logger(self.log_dir)
            policy.set_training_env(self.env)
            policy.use_demonstrations(self.demonstrations)
            policy.start_training()
            self.eval_policy_name_queue.put(name)
            if isinstance(policy, TD3Policy):
                web_app.web_globals._demonstrations_replay_buffer = policy.demonstrations_buffer
        self.cur_policy = name

    def names(self):
        return list(self.policies.keys())

    def __getitem__(self, item):
        return self.policies[item]

    def __contains__(self, item):
        return item in self.policies

    def __len__(self):
        return len(self.policies)
