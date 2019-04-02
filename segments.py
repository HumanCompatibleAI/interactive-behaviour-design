import glob
import os
import pickle
import time

import easy_tf_log

from rollouts import CompressedRollout
from utils import save_video, make_small_change


def prune_old_segments(dir, n_to_keep):
    pkl_files = glob.glob(os.path.join(dir, '*.pkl'))
    n_to_prune = len(pkl_files) - n_to_keep
    if n_to_prune <= 0:
        return
    pkl_files.sort(key=lambda fname: os.path.getmtime(fname))
    prune_pkl_files = pkl_files[:n_to_prune]
    prune_names = [os.path.basename(f).split('.')[0] for f in prune_pkl_files]
    for prune_name in prune_names:
        for prune_file in glob.glob(os.path.join(dir, prune_name + '.*')):
            os.remove(prune_file)


def monitor_segments_dir_loop(dir, n_to_keep):
    logger = easy_tf_log.Logger()
    logger.set_log_dir(dir)
    while True:
        time.sleep(30)
        prune_old_segments(dir, n_to_keep)
        n_segments = len(glob.glob(os.path.join(dir, '*.pkl')))
        logger.logkv('episode_segments/n_segments', n_segments)


def write_segments_loop(queue, dir):
    while True:
        obses, rewards, frames, episode_n = queue.get()
        frames = make_small_change(frames)
        segment = CompressedRollout(final_env_state=None,
                                    obses=obses,
                                    rewards=rewards,
                                    frames=frames,
                                    vid_filename=None,
                                    generating_policy=None,
                                    actions=None,
                                    extra_info=dict(episode_n=episode_n))
        base_name = os.path.join(dir, str(segment.hash))
        vid_filename = base_name + '.mp4'
        save_video(segment.frames, vid_filename)
        segment.vid_filename = os.path.basename(vid_filename)
        with open(base_name + '.pkl', 'wb') as f:
            pickle.dump(segment, f)