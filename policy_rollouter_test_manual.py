import glob
import multiprocessing
import os
import subprocess
import tempfile

import gym
import numpy as np
from flask import Flask, render_template, send_from_directory
from gym.wrappers import TimeLimit

import global_variables
from policy_rollouter import PolicyRollouter
from utils import configure_cpus
from wrappers.util_wrappers import ResetMode

env = gym.make('FetchReach-v1')  # type: TimeLimit
env._max_episode_seconds = None
env._max_episode_steps = None
reset_state_queue = multiprocessing.Queue()
reset_mode_value = multiprocessing.Value('i', ResetMode.USE_ENV_RESET.value)
global_variables.rollout_noise_sigma = 0.2
global_variables.rollout_mode = global_variables.RolloutMode.CUR_POLICY
global_variables.env_creation_lock = multiprocessing.Lock()
global_variables.rollout_randomness = global_variables.RolloutRandomness.CORRELATED_RANDOM_ACTION
global_variables.n_cur_policy = 4

script_dir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, template_folder=script_dir)
app.config['TEMPLATES_AUTO_RELOAD'] = True
vid_urls = []
rollout_hashes = []
noises = ['' for _ in range(4)]
policy_rollouter = None
temp_dir = tempfile.mkdtemp()
ckpt_dir = os.path.join(temp_dir, 'checkpoints')
demo_dir = os.path.join(temp_dir, 'demonstrations')
for d in [ckpt_dir, demo_dir]:
    os.makedirs(d)


class DummyPolicy:
    def step(self, obs):
        return np.zeros(env.action_space.shape)

    def load_checkpoint(self, ckpt_pass):
        return


def make_policy_fn(name, seed):
    return DummyPolicy()


@app.route('/')
def root_page():
    vid_glob = os.path.join(demo_dir, '*.mp4')

    for vid_path in glob.glob(vid_glob):
        os.remove(vid_path)
    for log in glob.glob(os.path.join(temp_dir, '*.log')):
        os.remove(log)

    policy_rollouter.generate_rollouts_from_reset(policies=['dummy'])

    vid_paths = glob.glob(vid_glob)
    for vid_path in vid_paths:
        vid_name = os.path.basename(vid_path)
        url = f'http://localhost:5000/videos/{vid_name}'
        vid_urls.append(url)

    for worker_n in [0, 1, 2, 3]:
        with open(os.path.join(temp_dir, f'actions_worker_{worker_n}.log'), 'r') as f:
            lines = f.read().rstrip().split('\n')
            first_5_lines = '\n'.join(lines[:5])
            last_5_lines = '\n'.join(lines[-5:])
            noises[worker_n] += first_5_lines + '\n...\n' + last_5_lines + '\n\n'

    return render_template('check_demonstration_rollouts.html',
                           vid1_url=vid_urls[0], vid2_url=vid_urls[1],
                           vid3_url=vid_urls[2], vid4_url=vid_urls[3],
                           vid5_url=vid_urls[4],
                           noise1=noises[0], noise2=noises[1],
                           noise3=noises[2], noise4=noises[3])


@app.route('/videos/<path:path>')
def send_video(path):
    return send_from_directory(demo_dir, path)


def main():
    global videos, rollout_hashes, policy_rollouter

    open(os.path.join(ckpt_dir, 'policy-dummy-0.meta'), 'w').close()
    configure_cpus(temp_dir, n_rollouter_cpus=1, n_drlhp_training_cpus=0)
    policy_rollouter = PolicyRollouter(env,
                                       save_dir=temp_dir,
                                       reset_state_queue_in=reset_state_queue,
                                       reset_mode_value=reset_mode_value,
                                       log_dir=temp_dir,
                                       make_policy_fn=make_policy_fn,
                                       redo_policy=False,
                                       noisy_policies=False,
                                       rollout_len_seconds=5,
                                       show_from_end_seconds=5,
                                       log_actions=True)

    subprocess.run(['open', 'http://localhost:5000'])
    app.run()


if __name__ == '__main__':
    main()
