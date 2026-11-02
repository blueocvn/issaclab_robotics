# Thu thap dataset HIL (DAgger) vong 2 bang checkpoint ACT fine-tune tren HIL_fine_tune_datasetlerobot
# Xoa thu muc dataset "rong" (chi co meta, khong co episode) do lan chay truoc de lai
D=/home/apc/demo/datasets/rollout_hil_so101_act_hil_r2
if [ -d "$D" ] && [ ! -d "$D/data" ] && [ ! -d "$D/videos" ]; then rm -r "$D"; fi
# Neu dataset da co episode thi nhay vao nhan tiep (resume), con lai tao moi
RESUME=false
if [ -d "$D/data" ]; then RESUME=true; fi
lerobot-rollout \
    --resume=$RESUME \
    --strategy.type=dagger \
    --policy.path=/home/apc/demo/outputs/act_hil_finetune_10k/checkpoints/last/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107630-if00 \
    --robot.id=so101_arm_latest \
    --robot.calibration_dir=/data/lerobot \
    --robot.cameras="{ Twist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110826-if00 \
    --teleop.id=so101_leader_latest \
    --teleop.calibration_dir=/data/lerobot \
    --dataset.repo_id=spectrecipher/rollout_hil_so101_act_hil_r2 \
    --dataset.root=/home/apc/demo/datasets/rollout_hil_so101_act_hil_r2 \
    --dataset.single_task="pick up box and put to white zone" \
    --dataset.push_to_hub=true \
    --task="pick up box and put to white zone" \
    --duration=60
