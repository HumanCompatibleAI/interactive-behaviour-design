import json
import os
import os.path as osp
import random
from collections import deque

import cv2
import numpy as np
import pims as pims
from PIL import Image
from flask import render_template, Blueprint
from flask import request, send_from_directory
from gym.wrappers.monitoring.video_recorder import ImageEncoder

from policies.td3 import SQIL_REWARD
from utils import save_video
from web_app.utils import nocache
from web_app.web_globals import _classifiers, _demonstrations_replay_buffer
from web_app.web_globals import _cur_label, FPS
from web_app.web_globals import _reward_switcher_wrapper
from web_app.web_globals import experience_dir
from web_app.web_globals import global_experience_buffer
from web_app.web_globals import save_dir

labelling_app = Blueprint('labelling', __name__)


@labelling_app.route('/label_video', methods=['GET'])
def label_video_frames():
    return render_template('label_video.html')


@labelling_app.route('/label_suggested', methods=['GET'])
def label_suggested_frames():
    return render_template('label_suggested.html')


@labelling_app.route('/tag', methods=['POST'])
def tag():
    post_data = json.loads(request.data)
    ep_name = post_data["epName"]
    label_name = post_data["labelName"]
    tag_type = post_data["tag"]

    if 'videoTime' in post_data:
        frame_idx = int(post_data['videoTime'] * FPS)
    elif 'frameIdx' in post_data:
        frame_idx = post_data['frameIdx']
    else:
        raise Exception("Frame reference not found in POST data")

    if tag_type == 'yes':
        label = 1
    elif tag_type == 'no':
        label = 0
    else:
        raise Exception("Unknown label {}".format(tag_type))

    global_experience_buffer.tag(ep_name, frame_idx, label_name, label)

    print("Current label count for this episode:",
          global_experience_buffer.label_counts(label_name, ep_name))
    print("Current label count for all episodes:",
          global_experience_buffer.label_counts(label_name))

    return ""


@labelling_app.route('/tag_goal_state', methods=['POST'])
def tag_goal_state():
    post_data = json.loads(request.data)
    ep_name = post_data["epName"]

    if 'videoTime' in post_data:
        frame_idx = int(post_data['videoTime'] * FPS)
    elif 'frameIdx' in post_data:
        frame_idx = post_data['frameIdx']
    else:
        raise Exception("Frame reference not found in POST data")

    obs = global_experience_buffer.episodes[ep_name].labelled_obses[frame_idx].obs
    # Unclear what we should set the action to.
    # We want to say: "Don't care about the action; just know that this state is really good."
    # So we could sample a bunch of random actions?
    act = None

    raise Exception('TODO: implement tag goal state')
    # _demonstrations_replay_buffer.store(
    #     obs=obs,
    #     act=act,
    #     rew=None,       # ignored
    #     next_obs=None,  # should be ignored with done=True
    #     done=True
    # )

    # return ""


@labelling_app.route('/sample_predictions', methods=['GET'])
def sample_predictions():
    if 'n' not in request.args:
        return render_template('sample_predictions.html')
    n = int(request.args['n'])

    if 'prediction' in request.args:
        prediction_filter = int(request.args['prediction'])
    else:
        prediction_filter = None

    frames = global_experience_buffer.get_all_obses()
    frames = random.sample(frames, 512)
    images = [frame.obs for frame in frames]
    preds_with_uncertainties = \
        _classifiers.predict_with_uncertainty(_cur_label, images)

    # batch_size = 512
    # n_batches = len(images) // batch_size
    # preds_with_uncertainties = []
    # for batch_n in range(n_batches):
    #     print("Batch {}/{}".format(1 + batch_n, n_batches))
    #     start = batch_n * batch_size
    #     end = start + batch_size
    #     batch = images[start:end]
    #     preds_with_uncertainties.extend(
    #         frames_classifier.predict_with_uncertainty(batch)
    #     )

    preds, uncertainties = zip(*preds_with_uncertainties)
    idxs_by_uncertainty = np.argsort(uncertainties)
    idxs_by_uncertainty = idxs_by_uncertainty[-n:]
    frames = [frames[i] for i in idxs_by_uncertainty]
    preds = [preds[i] for i in idxs_by_uncertainty]
    uncertainties = [uncertainties[i] for i in idxs_by_uncertainty]

    frames_with_info = []
    for frame, pred, uncertainty in zip(frames, preds, uncertainties):
        if prediction_filter is not None and pred != prediction_filter:
            continue
        im_url = make_frame_url(frame)
        frames_with_info.append((im_url,
                                 int(pred),
                                 "{:.2f}".format(uncertainty),
                                 frame.label,
                                 frame.validation))

    return json.dumps(frames_with_info)


@labelling_app.route('/get_label_list', methods=['GET'])
def get_label_list():
    return json.dumps(list(_classifiers.get_labels()))


@labelling_app.route('/get_episode_list', methods=['GET'])
def get_episode_list():
    episodes_with_videos = {ep_n: ep for ep_n, ep in global_experience_buffer.episodes.items()
                            if os.path.exists(ep.vid_path)}
    return json.dumps(list(episodes_with_videos.keys()))


def get_least_certain_frames(frames):
    images = [frame.obs for frame in frames]
    preds, confidences = zip(*_classifiers.predict_with_confidence(
        _cur_label, images))

    uncertainties = 1 - np.array(confidences)
    idxs_by_increasing_uncertainty = np.argsort(uncertainties)
    idxs_by_increasing_uncertainty = idxs_by_increasing_uncertainty[:10]
    frames = [frames[i] for i in idxs_by_increasing_uncertainty]

    return list(zip(frames, preds, confidences))


def get_most_confident_positive_frames(frames):
    images = [frame.obs for frame in frames]
    preds, confidences = zip(*_classifiers.predict_with_confidence(
        _cur_label, images))

    positive_pred_idxs = np.argwhere(np.array(preds) == 1).flatten()
    if len(positive_pred_idxs) == 0:
        raise Exception("No frames with positive predictions")
    frames = [frames[i] for i in positive_pred_idxs]
    preds = [preds[i] for i in positive_pred_idxs]
    confidences = [confidences[i] for i in positive_pred_idxs]

    idxs_by_increasing_confidence = np.argsort(confidences)
    most_confident_idxs = idxs_by_increasing_confidence[-10:]
    frames = [frames[i] for i in most_confident_idxs]
    preds = [preds[i] for i in most_confident_idxs]
    confidences = [confidences[i] for i in most_confident_idxs]

    return list(zip(frames, preds, confidences))


@labelling_app.route('/suggest_frame', methods=['GET'])
def suggest_frame():
    mode = request.args['mode']
    fresh = True if request.args['fresh'] == 'true' else False

    if fresh or not suggest_frame.frames:
        ep_name = global_experience_buffer.newest_episode()
        episode = global_experience_buffer.episodes[ep_name]
        frames = episode.sample(256, filter_obses_with_label='default')
        if mode == 'least_confident':
            frames = get_least_certain_frames(frames)
        elif mode == 'most_confident_positive':
            frames = get_most_confident_positive_frames(frames)
        else:
            raise Exception("Unknown frame suggestion mode:", mode)
        suggest_frame.frames = frames
        suggest_frame.ep_name = ep_name

    frame, pred, confidence = suggest_frame.frames.pop()
    im_url = make_frame_url(frame)

    return json.dumps((suggest_frame.ep_name,
                       im_url,
                       frame.timestep,
                       int(pred),
                       "{:.2f}".format(confidence)))


suggest_frame.frames = []
suggest_frame.ep_name = None


@labelling_app.route('/get_media', methods=['GET'])
@nocache
def get_media():
    filename = request.args['filename']
    return send_from_directory(save_dir, filename)


@labelling_app.route('/get_episode_vid', methods=['GET'])
@nocache
def get_episode_vid():
    ep_n = request.args['ep_n']
    episode = global_experience_buffer.episodes[ep_n]

    dir_name, vid_name = os.path.dirname(episode.vid_path), os.path.basename(episode.vid_path)
    if 'rewards' not in request.args and 'predictions' not in request.args:
        return send_from_directory(dir_name, vid_name)

    suffix = ''
    if 'rewards' in request.args:
        suffix += '-rewards'
    if 'predictions' in request.args:
        suffix += '-predictions'
    vid_name = vid_name.replace('.mp4', '{}.mp4'.format(suffix))
    if vid_name in os.listdir(dir_name) and request.args['cached_ok'] == 'true':
        return send_from_directory(dir_name, vid_name)

    frames = list(pims.Video(episode.vid_path))
    obses = [lobs.obs for lobs in episode.labelled_obses]
    if 'rewards' in request.args:
        render_drlhp_reward(obses, frames)
    if 'predictions' in request.args:
        render_probs(obses, frames)
    save_video(frames, os.path.join(dir_name, vid_name))
    return send_from_directory(dir_name, vid_name)


def make_frame_url(frame):
    im = frame.obs[:, :, -1]
    im = im[:, :, None]
    im = im.repeat(3, axis=2)
    im = Image.fromarray(im)
    filename = "{}.png".format(frame.timestep)
    im.save(osp.join(save_dir, filename))
    im_url = "http://127.0.0.1:5000/get_media?filename=" + filename
    return im_url


def render_probs(obses, images):
    probs = {}
    labels = _classifiers.get_labels()
    for label_name in labels:
        probs[label_name] = _classifiers.predict_positive_prob(label_name, obses)
    for image_n, image in enumerate(images):
        for label_n, label_name in enumerate(labels):
            prob = probs[label_name][image_n]
            color = [int(255.0 * prob)] * images[0].shape[-1]
            cv2.putText(image,
                        label_name,
                        org=(15, 15 + label_n * 10),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=0.5,
                        color=color,
                        thickness=1)


def render_drlhp_reward(obses, images):
    rews = _reward_switcher_wrapper.reward_predictor.reward(np.array(obses))
    im_width = images[0].shape[1]
    graph = deque(maxlen=(im_width - 10))
    for image_n, image in enumerate(images):
        graph.append(rews[image_n])
        image[30, 5:-5, :] = 255
        image[20, 5:-5, :] = 255
        image[10, 5:-5, :] = 255
        image[10:30, 5, :] = 255
        image[10:30, -5, :] = 255
        for x, val in enumerate(graph):
            y = int(val / max(np.abs(rews)) * 10)
            image[20 - y, 5 + x, :] = 255

        cv2.putText(image,
                    "{:.3f}".format(rews[image_n]),
                    org=(20, 50),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=0.5,
                    color=[255] * images[0].shape[-1],
                    thickness=1)


def render_rollout(filename, rollout_frames):
    vid_path = os.path.join(experience_dir, filename)
    encoder = None
    for frame in rollout_frames:
        im = np.tile(frame[:, :, -1][:, :, None], [1, 1, 3])
        if encoder is None:
            encoder = ImageEncoder(vid_path, im.shape, frames_per_sec=30)
        encoder.capture_frame(im)
    encoder.close()
