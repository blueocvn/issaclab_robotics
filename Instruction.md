# Instruction: Create a Dataset and Train a Model (SO-101)

Two ways to do the same job:

- **Part A:** the LeLab GUI.
- **Part B:** the terminal, with `lerobot-record` and `lerobot-train`.

Either way, the result is a LeRobot dataset, then a trained ACT policy, which you test with the rollout scripts in this repo.

## 0. Prerequisites

1. Connect the **leader** arm, the **follower** arm and **2 cameras** (`Twist`, `top`) over USB.
2. Activate the environment:
   ```bash
   cd /data/lerobot
   source .venv/bin/activate
   ```
3. Log in to Hugging Face. You need this to push the dataset or model:
   ```bash
   huggingface-cli login
   ```
4. Find your ports and cameras. Prefer the stable `by-id` paths:
   ```bash
   ls /dev/serial/by-id/          # arm serial ports
   lerobot-find-port              # or: unplug/replug to identify a port
   lerobot-find-cameras opencv    # camera indices
   ```
5. Check that the **calibration files** exist in `/data/lerobot`:
   - Follower: `so101_arm_latest.json`. The repo currently has only `so101_arm.json`, so copy or rename it.
   - Leader: `so101_leader_latest.json`. It is not in the repo right now.

   Look for them in `~/.cache/huggingface/lerobot/calibration` and copy them to `/data/lerobot`. If they are missing, or the arms were re-assembled, re-calibrate (see A2 or B1).

---

## Part A: LeLab GUI

### A1. Start LeLab and connect the robot
1. Start LeLab (the command depends on how you installed it) and open the web UI in your browser.
2. Select the **follower port** and **leader port**.
3. Add the two cameras. Set the names to **`Twist`** and **`top`**, 640x480, 30 fps. The camera names become the policy's input names, so keep them consistent. `Twist` is rotated 180°.
4. Save. This writes a robot config like [so101_arm_robot_config.json](so101_arm_robot_config.json).

### A2. Calibrate (first time, or after any hardware change)
1. Open **Calibration** and choose the follower, then the leader.
2. Follow the on-screen steps. Move each joint through its full range and put the arm in the middle position when asked.
3. Save. The calibration JSON files are written to the calibration folder.

### A3. Check teleoperation
1. Open **Teleoperation** and move the leader arm.
2. The follower should mirror it smoothly, and both camera views should be live.
3. If it does not, go back to A1 and A2.

### A4. Record the dataset
1. Open **Recording** and fill in:
   | Field | Example |
   |---|---|
   | Dataset repo id | `spectrecipher/my_dataset` |
   | Task description | `pick up box and put to white zone` |
   | Number of episodes | 30-50 to start |
   | Episode length | 20-30 s |
   | Reset time | 5-10 s (put the box back) |
   | FPS | 30 |
   | Push to Hub | on, if you want a backup |
2. Click **Start recording**.
3. For each episode, teleoperate the whole task with the leader arm. Use the reset period to put the box back.
4. Use the on-screen controls (or keyboard shortcuts) to **save**, **re-record** or **stop** an episode.
5. When finished, the dataset is stored locally (by default under `~/.cache/huggingface/lerobot/<repo_id>`) and optionally pushed to the Hub.

**Tips for good data**
- Vary the box position a little between episodes.
- Keep the lighting and camera positions the same as at test time.
- Make each demo clean and consistent. Avoid failed attempts and long pauses.
- Aim for at least 30 good episodes.

### A5. Train
1. Open **Training**.
2. Choose:
   | Field | Value |
   |---|---|
   | Dataset | the repo id from A4 |
   | Policy | `act` |
   | Steps | 10000 (as a start) |
   | Batch size | 8 (lower it if the GPU runs out of memory) |
   | Device | `cuda` |
3. Click **Start training** and watch the loss curve. The checkpoints are saved under the output folder (`.../checkpoints/<step>/pretrained_model`, plus `last`).

### A6. Test
Go on to **Section C** (rollout).

---

## Part B: Terminal

### B1. Calibrate (first time, or after any hardware change)
```bash
# Follower
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107630-if00 \
  --robot.id=so101_arm_latest \
  --robot.calibration_dir=/data/lerobot

# Leader
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110826-if00 \
  --teleop.id=so101_leader_latest \
  --teleop.calibration_dir=/data/lerobot
```
Follow the prompts. The result is `<id>.json` in `/data/lerobot`.

### B2. Check teleoperation
```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82107630-if00 \
  --robot.id=so101_arm_latest \
  --robot.calibration_dir=/data/lerobot \
  --robot.cameras="{ Twist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5C82110826-if00 \
  --teleop.id=so101_leader_latest \
  --teleop.calibration_dir=/data/lerobot \
  --display_data=true
```

### B3. Record the dataset
This is the content of [create_dataset.sh](create_dataset.sh). Edit `--dataset.repo_id` and `--dataset.root` to a new name, then run `bash create_dataset.sh`:
```bash
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
```
Keyboard controls while recording:

| Key | Action |
|---|---|
| `→` | End the current episode (or the reset period) early |
| `←` | Discard and re-record the current episode |
| `Esc` | Stop the session |

To add episodes to an existing dataset, run the same command with `--resume=true`.

Sanity check afterwards:
```bash
ls /data/lerobot/datasets/my_dataset     # expect: data/  meta/  videos/
```
Preview it at https://huggingface.co/spaces/lerobot/visualize_dataset (needs the dataset on the Hub).

### B4. Train
**From scratch**, with a new ACT policy:
```bash
cd /data/lerobot
.venv/bin/lerobot-train \
  --policy.type=act \
  --dataset.repo_id=spectrecipher/my_dataset \
  --dataset.root=/data/lerobot/datasets/my_dataset \
  --steps=10000 --batch_size=8 --save_freq=1000 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --job_name=act_my_dataset_10k \
  --output_dir=/data/lerobot/outputs/act_my_dataset_10k
```

**Fine-tune** an existing checkpoint, for example on new HIL data. This is what [train.sh](train.sh) does:
```bash
.venv/bin/lerobot-train \
  --policy.path=<path/to/checkpoint>/pretrained_model \
  --dataset.repo_id=spectrecipher/HIL_fine_tune_datasetlerobot_r2 \
  --dataset.root=/home/apc/demo/datasets/HIL_fine_tune_datasetlerobot_r2 \
  --steps=10000 --batch_size=8 --save_freq=1000 \
  --policy.push_to_hub=false --wandb.enable=false \
  --job_name=act_hil_finetune_r2b_10k \
  --output_dir=/home/apc/demo/outputs/act_hil_finetune_r2b_10k
```

Notes:
- `--output_dir` must not already exist, or training will refuse to start. Use a new name or delete the old folder.
- Turn on Weights & Biases with `--wandb.enable=true` (after `wandb login`).
- If you hit CUDA out-of-memory, lower `--batch_size`.
- Checkpoints are at `<output_dir>/checkpoints/<step>/pretrained_model`, and `last` points to the newest.

---

## C. Test the trained model (rollout)

1. Set `--policy.path` in [rollout.sh](rollout.sh) to your checkpoint, either a local `.../checkpoints/last/pretrained_model` or a Hub id. For example:
   ```
   --policy.path=/data/lerobot/outputs/act_my_dataset_10k/checkpoints/last/pretrained_model
   ```
2. Clear the workspace, put the box in place, and keep a hand near the power or e-stop.
3. Run:
   ```bash
   bash rollout.sh
   ```
4. If the policy fails in some situations, collect corrections with DAgger / HIL ([rollout_dagger_hil_r1.sh](rollout_dagger_hil_r1.sh), [rollout_dagger_hil_r2.sh](rollout_dagger_hil_r2.sh)), then fine-tune with `train.sh`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Permission denied` on `/dev/ttyACM0` | `sudo usermod -aG dialout $USER`, then log out and back in. |
| Wrong camera shows up | Re-run `lerobot-find-cameras opencv` and fix `index_or_path`. The indices can change after a reboot. |
| Arm jumps or moves wrongly | The calibration is wrong or does not match the `--robot.id`. Re-calibrate. |
| `Output directory already exists` | Use a new `--output_dir`, or `--resume=true` to continue the run. |
| Policy shakes or does nothing | Check that camera names and positions match the training data. Record more varied demos. |
| Push to Hub fails | Run `huggingface-cli login` and check that the repo id uses your username. |

---

## Reference: samples from the Hub

Use these as working examples. The full list is in the README, section 5.

- Models: <https://huggingface.co/spectrecipher/models>
- Datasets: <https://huggingface.co/spectrecipher/datasets>

**Sample dataset to copy the format from or train on:** [spectrecipher/dataset_new_calib_20261028_085359](https://huggingface.co/datasets/spectrecipher/dataset_new_calib_20261028_085359) (61 episodes). It is the dataset behind the 10k-step base model in `pretrained_model/`. You can train directly on it without recording:
```bash
.venv/bin/lerobot-train \
  --policy.type=act \
  --dataset.repo_id=spectrecipher/dataset_new_calib_20261028_085359 \
  --steps=10000 --batch_size=8 --save_freq=1000 \
  --policy.push_to_hub=false --wandb.enable=false \
  --job_name=act_sample_10k \
  --output_dir=/data/lerobot/outputs/act_sample_10k
```
Training on it works only if your cameras are named `Twist` and `top`, and the arm calibration is compatible.

**Sample trained policies:** run one without training anything by pointing `--policy.path` at a Hub id, as [rollout_dagger.sh](rollout_dagger.sh) does:
```
--policy.path=spectrecipher/so101-act-demo-v2
--policy.path=spectrecipher/act_hil_finetune_10k
```

**What I did before (the history of this project):**

| Round | Data | Result |
|---|---|---|
| 1. Demos | `demo_datasetDemo_20261026_190035` (81 ep), then `dataset_new_calib_20261028_085359` (61 ep) after re-calibration | ACT models `so101-act-demo-*` and the 10k base model |
| 2. HIL round 1 | `rollout_hil_so101_act_10k` (22 ep) | `HIL_fine_tune_datasetlerobot` (83 ep) and `act_hil_finetune_10k` |
| 3. HIL round 2 | `rollout_hil_so101_act_hil_r2` (21 ep) | `HIL_fine_tune_datasetlerobot_r2` (104 ep), trained with `train.sh` |

Dataset naming convention used so far: `demo_*` or `dataset_*` for teleoperated demos, `rollout_hil_*` for DAgger/HIL corrections, and `HIL_fine_tune_*` for the combined set used to fine-tune. Reuse it so the repos stay easy to tell apart.
