import glob
import os
import pickle
import shutil
import time

import easy_tf_log

import global_variables
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
            if not global_variables.render_segments:
                os.remove(prune_file)
            else:
                old_segs_dir = os.path.join(dir, 'old_segs')
                os.makedirs(old_segs_dir, exist_ok=True)
                # It's unlikely we'll get two segments with the same ID, but not impossible
                dst = os.path.join(old_segs_dir, os.path.basename(prune_file))
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(prune_file, dst)


def monitor_segments_dir_loop(dir, n_to_keep):
    logger = easy_tf_log.Logger()
    logger.set_log_dir(dir)
    while True:
        time.sleep(5)
        prune_old_segments(dir, n_to_keep)
        n_segments = len(glob.glob(os.path.join(dir, '*.pkl')))
        logger.logkv('episode_segments/n_segments', n_segments)


def write_segments_loop(queue, dir):
    while True:
        obses, rewards, frames = queue.get()
        frames = make_small_change(frames)
        segment = CompressedRollout(final_env_state=None,
                                    obses=obses,
                                    rewards=rewards,
                                    frames=frames,
                                    vid_filename=None,
                                    generating_policy=None,
                                    actions=None)
        base_name = os.path.join(dir, str(segment.hash))
        vid_filename = base_name + '.mp4'
        save_video(segment.frames, vid_filename)
        segment.vid_filename = os.path.basename(vid_filename)
        with open(base_name + '.pkl', 'wb') as f:
            pickle.dump(segment, f)
