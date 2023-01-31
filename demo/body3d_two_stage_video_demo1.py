import copy
import os
import cv2
import mmcv
import numpy as np

from mmpose.apis import (get_track_id, inference_pose_lifter_model,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_3d_pose_result)
from lib.dataset_info import DatasetInfo
from mmdet.apis import inference_detector, init_detector


def extract_pose_sequence(pose_results, frame_idx, seq_len, step=1):

    frames_left = (seq_len - 1) // 2
    frames_right = frames_left
    num_frames = len(pose_results)

    # get the padded sequence
    pad_left = max(0, frames_left - frame_idx // step)
    pad_right = max(0, frames_right - (num_frames - 1 - frame_idx) // step)
    start = max(frame_idx % step, frame_idx - frames_left * step)
    end = min(num_frames - (num_frames - 1 - frame_idx) % step, frame_idx + frames_right * step + 1)
    pose_results_seq = [pose_results[0]] * pad_left + pose_results[start:end:step] + [pose_results[-1]] * pad_right
    return pose_results_seq


def convert_keypoint_definition(keypoints):

    keypoints_new = np.zeros((17, keypoints.shape[1]), dtype=keypoints.dtype)
    keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
    keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
    keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
    keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
    keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
        keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]

    return keypoints_new


def pose_processing(pose_lift_model, pose_lift_dataset,
                    pose_lift_dataset_info, pose_det_results, config, resolution, pose_det_results_list, i, data_cfg):

    pose_results_2d = extract_pose_sequence(
        pose_det_results_list,
        frame_idx=i,
        seq_len=data_cfg.seq_len,
        step=data_cfg.seq_frame_interval)

    pose_lift_results = inference_pose_lifter_model(
        pose_lift_model,
        pose_results_2d=pose_results_2d,
        dataset=pose_lift_dataset,
        dataset_info=pose_lift_dataset_info,
        with_track_id=True,
        image_size=resolution,
        norm_pose_2d=config["norm_pose_2d"])

    pose_lift_results_vis = []
    for idx, res in enumerate(pose_lift_results):
        keypoints_3d = res['keypoints_3d']
        # exchange y,z-axis, and then reverse the direction of x,z-axis
        keypoints_3d = keypoints_3d[..., [0, 2, 1]]
        keypoints_3d[..., 0] = -keypoints_3d[..., 0]
        keypoints_3d[..., 2] = -keypoints_3d[..., 2]
        # rebase height (z-axis)
        if config["rebase_keypoint_height"]:
            keypoints_3d[..., 2] -= np.min(
                keypoints_3d[..., 2], axis=-1, keepdims=True)
        res['keypoints_3d'] = keypoints_3d
        # add title
        det_res = pose_det_results[idx]
        instance_id = det_res['track_id']
        res['title'] = f'Prediction ({instance_id})'
        # only visualize the target frame
        res['keypoints'] = det_res['keypoints']
        res['bbox'] = det_res['bbox']
        res['track_id'] = instance_id
        pose_lift_results_vis.append(res)
    return pose_lift_results_vis


def main():
    import yaml
    with open("demo/config_video.yaml") as file:
        config = yaml.safe_load(file)

    video = mmcv.VideoReader(config["video_path"])

    print('Stage 1: 2D pose detection.')

    print('Initializing model...')
    person_det_model = init_detector(config["det_config"], config["det_checkpoint"], device=config["device"])
    pose_det_model = init_pose_model(config["pose_detector_config"], config["pose_detector_checkpoint"], device=config["device"])

    pose_det_dataset = pose_det_model.cfg.data['test']['type']
    dataset_info = pose_det_model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)

    pose_det_results_list = []
    next_id = 0
    pose_det_results = []

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    print('Running 2D pose detection inference...')

    pose_lift_model = init_pose_model(
        config["pose_lifter_config"],
        config["pose_lifter_checkpoint"],
        device=config["device"])

    pose_lift_dataset = pose_lift_model.cfg.data['test']['type']

    if config["out_video_root"] == '':
        save_out_video = False
    else:
        os.makedirs(config["out_video_root"], exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video.fps
        writer = None

    # load temporal padding config from model.data_cfg
    if hasattr(pose_lift_model.cfg, 'test_data_cfg'):
        data_cfg = pose_lift_model.cfg.test_data_cfg
    else:
        data_cfg = pose_lift_model.cfg.data_cfg

    num_instances = config["num_instances"]
    pose_lift_dataset_info = pose_lift_model.cfg.data['test'].get('dataset_info', None)
    pose_lift_dataset_info = DatasetInfo(pose_lift_dataset_info)

    # START
    for cur_frame in mmcv.track_iter_progress(video):
        pose_det_results_last = pose_det_results

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(person_det_model, cur_frame)

        # keep the person class bounding boxes.
        person_det_results = process_mmdet_results(mmdet_results, config["det_cat_id"])

        # make person results for current image
        pose_det_results, _ = inference_top_down_pose_model(
            pose_det_model,
            cur_frame,
            person_det_results,
            bbox_thr=config["bbox_thr"],
            format='xyxy',
            dataset=pose_det_dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=output_layer_names)

        # get track id for each person instance
        pose_det_results, next_id = get_track_id(
            pose_det_results,
            pose_det_results_last,
            next_id,
            use_oks=config["use_oks_tracking"],
            tracking_thr=config["tracking_thr"])

        pose_det_results_list.append(copy.deepcopy(pose_det_results))

    # convert keypoint definition
    for pose_det_results in pose_det_results_list:
        for res in pose_det_results:
            keypoints = res['keypoints']
            res['keypoints'] = convert_keypoint_definition(keypoints)

    for i, pose_det_results in enumerate(mmcv.track_iter_progress(pose_det_results_list)):
        # extract and pad input pose2d sequence
        # 2D-to-3D pose lifting
        # Pose processing
        pose_lift_results_vis = pose_processing(pose_lift_model, pose_lift_dataset,
                    pose_lift_dataset_info, pose_det_results, config, video.resolution, pose_det_results_list, i, data_cfg)

        # Visualization
        if num_instances < 0:
            num_instances = len(pose_lift_results_vis)
        img_vis = vis_3d_pose_result(
            pose_lift_model,
            result=pose_lift_results_vis,
            img=video[i],
            dataset=pose_lift_dataset,
            dataset_info=pose_lift_dataset_info,
            out_file=None,
            radius=config["radius"],
            thickness=config["thickness"],
            num_instances=num_instances,
            show=config["show"])

        if save_out_video:
            if writer is None:
                writer = cv2.VideoWriter(
                    os.path.join(config["out_video_root"],
                             f'vis_{os.path.basename(config["video_path"])}'), fourcc,
                    fps, (img_vis.shape[1], img_vis.shape[0]))
            writer.write(img_vis)

    if save_out_video:
        writer.release()


if __name__ == '__main__':
    main()
