python demo/body3d_two_stage_video_demo.py `
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py `
    ~/.cache/torch/hub/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth `
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py `
    ~/.cache/torch/hub/checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth `
    configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py `
    ~/.cache/torch/hub/checkpoints/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth `
    --video-path tests/data/h36m/sample.mp4 `
    --out-video-root vis_results `
    --rebase-keypoint-height