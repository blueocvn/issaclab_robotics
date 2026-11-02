# Thu thap dataset HIL (DAgger) bang checkpoint ACT 10000 step moi train
# Xoa thu muc dataset "rong" (chi co meta, khong co episode) do lan chay truoc de lai
D=/data/lerobot/datasets/rollout_hil_so101_act_10k
if [ -d "$D" ] && [ ! -d "$D/data" ] && [ ! -d "$D/videos" ]; then rm -r "$D"; fi
# Neu dataset da co episode thi nhay vao nhan tiep (resume), con lai tao moi
RESUME=false
if [ -d "$D/data" ]; then RESUME=true; fi
lerobot-rollout \
    --resume=$RESUME \
    --strategy.type=dagger \
    --policy.path=/home/apc/.cache/huggingface/lerobot/outputs/train/act_spectrecipher_dataset_new_calib_20261028_085359_2026-10-28_10-05-32/run/checkpoints/last/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107630-if00 \
    --robot.id=so101_arm_latest \
    --robot.calibration_dir=/data/lerobot \
    --robot.cameras="{ Twist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110826-if00 \
    --teleop.id=so101_leader_latest \
    --teleop.calibration_dir=/data/lerobot \
    --dataset.repo_id=spectrecipher/rollout_hil_so101_act_10k \
    --dataset.root=/data/lerobot/datasets/rollout_hil_so101_act_10k \
    --dataset.single_task="pick up box and put to white zone" \
    --dataset.push_to_hub=true \
    --task="pick up box and put to white zone" \
    --duration=60
