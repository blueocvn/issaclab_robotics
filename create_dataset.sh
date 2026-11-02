cd /data/lerobot
.venv/bin/lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107630-if00 \
  --robot.id=so101_arm_latest \
  --robot.calibration_dir=/data/lerobot \
  --robot.cameras="{ Twist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110826-if00 \
  --teleop.id=so101_leader_latest \
  --teleop.calibration_dir=/data/lerobot \
  --dataset.repo_id=spectrecipher/my_dataset \
  --dataset.root=/data/lerobot/datasets/my_dataset \
  --dataset.single_task="pick up box and put to white zone" \
  --dataset.num_episodes=30 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --dataset.fps=30 \
  --dataset.push_to_hub=true \
  --display_data=true
