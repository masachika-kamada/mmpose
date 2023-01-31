import os
import warnings
import cv2
import mmcv
import numpy as np
import glob

from mmpose.apis import (inference_pose_lifter_model,
                         inference_top_down_pose_model, vis_3d_pose_result,
                         init_pose_model, process_mmdet_results)
from mmpose.apis.inference import init_pose_model
from mmpose.core.camera import SimpleCamera
from mmpose.datasets import DatasetInfo
from mmdet.apis import inference_detector, init_detector


def convert_keypoint_definition(keypoints):
    keypoints_new = np.zeros((17, keypoints.shape[1]), dtype=keypoints.dtype)
    keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
    keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
    keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
    keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
    keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
        keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]

    return keypoints_new


def _keypoint_camera_to_world(keypoints, camera_params, si, di):
    # cam_key = None
    # print("\nimage_name", image_name)
    # subj, rest = os.path.basename(image_name).split("_", 1)
    # _, rest = rest.split(".", 1)
    # camera, rest = rest.split("_", 1)
    # cam_key = (subj, camera)

    # print("cam_key", cam_key)
    # camera = SimpleCamera(camera_params[cam_key])
    # print("camera_params", camera_params.keys())
    camera = SimpleCamera(camera_params[(f'S{si}', di)])
    keypoints_world = keypoints.copy()
    keypoints_world[..., :3] = camera.camera_to_world(keypoints[..., :3])

    return keypoints_world


def main(si, di):
    import yaml
    with open("demo/config_img.yaml") as file:
        config = yaml.safe_load(file)

    # First stage: 2D pose detection
    pose_det_results_list = []
    print("Stage 1: 2D pose detection.")

    pose_det_model = init_pose_model(
        config["pose_detector_config"],
        config["pose_detector_checkpoint"],
        device=config["device"])

    dataset = pose_det_model.cfg.data["test"]["type"]
    dataset_info = pose_det_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn(
            "Please set `dataset_info` in the config."
            "Check https://github.com/open-mmlab/mmpose/pull/663 "
            "for details.", DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    dir_ref = config["img_root"]
    person_det_model = init_detector(config["det_config"], config["det_checkpoint"], device=config["device"])
    pose_det_dataset = pose_det_model.cfg.data["test"]["type"]
    for image_name in mmcv.track_iter_progress(glob.glob(dir_ref + "/*.jpg")):
        # get bounding box annotations
        frame = cv2.imread(image_name)
        mmdet_results = inference_detector(person_det_model, frame)

        # keep the person class bounding boxes.
        person_det_results = process_mmdet_results(mmdet_results, config["det_cat_id"])

        # make person results for current image
        pose_det_results, _ = inference_top_down_pose_model(
            pose_det_model,
            frame,
            person_det_results,
            bbox_thr=config["bbox_thr"],
            format="xyxy",
            dataset=pose_det_dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None)

        # make person results for single image
        person_results = []
        # img_copy = frame.copy()
        for res in pose_det_results:
            person = {}
            person["bbox"] = res["bbox"]
            person_results.append(person)
        #     # bboxを描画
        #     cv2.rectangle(img_copy, (int(res["bbox"][0]), int(res["bbox"][1])), (int(res["bbox"][2]), int(res["bbox"][3])), (0, 0, 255), 2)
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow("img", img_copy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        pose_det_results, _ = inference_top_down_pose_model(
            pose_det_model,
            image_name,
            person_results,
            bbox_thr=None,
            format="xywh",
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=False,
            outputs=None)

        for res in pose_det_results:
            res["image_name"] = image_name
        pose_det_results_list.append(pose_det_results)

    # convert keypoint definition
    for pose_det_results in pose_det_results_list:
        for res in pose_det_results:
            keypoints = res["keypoints"]
            res["keypoints"] = convert_keypoint_definition(keypoints)

    # Second stage: Pose lifting
    print("Stage 2: 2D-to-3D pose lifting.")

    pose_lift_model = init_pose_model(
        config["pose_lifter_config"],
        config["pose_lifter_checkpoint"],
        device=config["device"])

    dataset_info = pose_lift_model.cfg.data["test"].get("dataset_info", None)
    if dataset_info is None:
        warnings.warn(
            "Please set `dataset_info` in the config."
            "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    camera_params = None
    if config["camera_param_file"] is not None:
        camera_params = mmcv.load(config["camera_param_file"])

    for i, pose_det_results in enumerate(mmcv.track_iter_progress(pose_det_results_list)):
        # 2D-to-3D pose lifting
        # Note that the pose_det_results are regarded as a single-frame pose
        # sequence
        pose_lift_results = inference_pose_lifter_model(
            pose_lift_model,
            pose_results_2d=[pose_det_results],
            dataset=dataset,
            dataset_info=dataset_info,
            with_track_id=False)

        image_name = pose_det_results[0]["image_name"]

        # Pose processing
        pose_lift_results_vis = []
        for idx, res in enumerate(pose_lift_results):
            keypoints_3d = res["keypoints_3d"]
            # project to world space
            if camera_params is not None:
                keypoints_3d = _keypoint_camera_to_world(keypoints_3d, camera_params, si, di)
            # rebase height (z-axis)
            if config["rebase_keypoint_height"]:
                keypoints_3d[..., 2] -= np.min(
                    keypoints_3d[..., 2], axis=-1, keepdims=True)
            res["keypoints_3d"] = keypoints_3d
            # Add title
            det_res = pose_det_results[idx]
            instance_id = det_res.get("track_id", idx)
            res["title"] = f"Prediction ({instance_id})"
            pose_lift_results_vis.append(res)
            # Add ground truth
            if config["show_ground_truth"]:
                if "keypoints_3d" not in det_res:
                    print("Fail to show ground truth. Please make sure that"
                          " the instance annotations from the Json file"
                          " contain 'keypoints_3d'.")
                else:
                    gt = res.copy()
                    gt["keypoints_3d"] = det_res["keypoints_3d"]
                    gt["title"] = f"Ground truth ({instance_id})"
                    pose_lift_results_vis.append(gt)

        # Visualization
        if config["out_img_root"] is None:
            out_file = None
        else:
            os.makedirs(config["out_img_root"], exist_ok=True)
            out_file = os.path.join(config["out_img_root"], f"vis_{i}_{si}_{di}.jpg")

        # opencvで描画しているだけ
        vis_3d_pose_result(
            pose_lift_model,
            result=pose_lift_results_vis,
            img=image_name,
            dataset_info=dataset_info,
            out_file=out_file)
        # out_fileに頭の高さを追加で書き込む
        if out_file is not None:
            img_tmp = cv2.imread(out_file)
            h, w = img_tmp.shape[:2]
            info = f"head: {pose_lift_results_vis[0]['keypoints_3d'][10][2]:.3f} m"
            cv2.putText(img_tmp, info, (w - 300, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(out_file, img_tmp)


if __name__ == "__main__":
    for si in range(1, 12):
        for di in ['54138969', '55011271', '58860488', '60457274']:
            main(si, di)
