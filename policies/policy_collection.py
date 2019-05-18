from typing import Dict

from policies.base_policy import Policy
from rollouts import RolloutsByHash


class PolicyCollection:
    policies: Dict[str, Policy]

    def __init__(self, make_policy_fn, log_dir, demonstrations: RolloutsByHash, seed):
        self.policies = {}
        self.cur_policy = None
        self.make_policy = make_policy_fn
        self.env = None
        self.log_dir = log_dir
        self.demonstrations = demonstrations
        self.seed = seed

    def add_policy(self, name, policy_kwargs=None):
        policy_kwargs = policy_kwargs or {}
        policy_kwargs.update({'seed': self.seed})
        self.policies[name] = self.make_policy(name, **policy_kwargs)
        print(f"Added policy '{name}'")

    def set_active_policy(self, name):
        for policy in self.policies.values():
            policy.stop_training()
        if name is not None:
            self.policies[name].init_logger(self.log_dir)
            self.policies[name].set_training_env(self.env, self.log_dir)
            self.policies[name].use_demonstrations(self.demonstrations)
            self.policies[name].start_training()
        self.cur_policy = name

    def names(self):
        return list(self.policies.keys())

    def __getitem__(self, item):
        return self.policies[item]

    def __contains__(self, item):
        return item in self.policies

    def __len__(self):
        return len(self.policies)
