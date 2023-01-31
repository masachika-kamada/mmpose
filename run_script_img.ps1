python demo/body3d_two_stage_img_demo.py `
    configs/body/3d_kpt_sview_rgb_img/pose_lift/h36m/simplebaseline3d_h36m.py `
    ~/.cache/torch/hub/checkpoints/simple3Dbaseline_h36m-f0ad73a4_20210419.pth `
    --json-file tests/data/h36m/h36m_coco.json `
    --img-root tests/data/h36m `
    --camera-param-file tests/data/h36m/cameras.pkl `
    --only-second-stage `
    --out-img-root vis_results `
    --rebase-keypoint-height `
    --show-ground-truth
