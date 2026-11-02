lerobot-rollout \
    --strategy.type=base \
    --policy.path=/home/apc/demo/outputs/act_hil_finetune_r2b_10k/checkpoints/last/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=so101_arm_latest \
    --robot.calibration_dir=/data/lerobot \
    --robot.cameras="{ Twist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --task="pick up box and put to white zone" \
    --duration=30
