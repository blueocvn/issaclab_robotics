# SO-101 Arm: Imitation Learning with LeLab + LeRobot

This repo holds the files for teaching an **SO-101** robot arm to *"pick up box and put to white zone"* with **imitation learning**. It covers calibration, a trained ACT policy, and the scripts for training, rollout and DAgger / HIL (human-in-the-loop) data collection. To record your own data and train, start with [Instruction.md](Instruction.md).

## 1. Frameworks

### LeRobot
[LeRobot](https://github.com/huggingface/lerobot) is Hugging Face's open-source PyTorch library for real-world robotics. It provides:

- Robot and teleoperator drivers (here `so101_follower` and `so101_leader`).
- The `LeRobotDataset` format, with Parquet data and MP4 video, which can be shared on the Hugging Face Hub.
- Policies such as ACT, Diffusion Policy and SmolVLA.
- CLI tools: `lerobot-record`, `lerobot-train` and `lerobot-rollout`.

### LeLab
**LeLab** is a web UI on top of LeRobot. It replaces most of the command-line steps with a browser interface for:

| Step | What you do in LeLab |
|---|---|
| Setup | Pick the leader and follower serial ports and the cameras. This is saved as [so101_arm_robot_config.json](so101_arm_robot_config.json). |
| Calibration | Calibrate the leader and follower arms. This produces the calibration JSON files in this repo. |
| Teleoperation | Drive the follower with the leader arm. |
| Recording | Record demonstration episodes into a LeRobot dataset, optionally pushed to the Hub. |
| Training | Launch policy training on a recorded dataset. |

LeLab handles the hardware side and data collection. The scripts in this repo cover the loop after that: fine-tuning, rollout and DAgger.

## 2. Imitation learning workflow

```
 ┌─────────────┐   ┌────────────┐   ┌───────────┐   ┌──────────┐
 │ Calibrate   │──▶│ Teleop     │──▶│ Train ACT │──▶│ Rollout  │
 │ (LeLab)     │   │ demos      │   │ policy    │   │ on robot │
 └─────────────┘   │ (LeLab)    │   └───────────┘   └────┬─────┘
                   └────────────┘         ▲              │
                                          │   ┌──────────▼───────────┐
                                          └───│ DAgger / HIL:        │
                                              │ human corrects the   │
                                              │ policy, new episodes │
                                              └──────────────────────┘
```

1. **Calibrate** the leader and follower arms in LeLab.
2. **Record demonstrations** by teleoperating with the leader arm. Two cameras are used, `Twist` and `top`, at 640x480 and 30 fps.
3. **Train** an ACT (Action Chunking with Transformers) policy with `lerobot-train`. The base model was trained for 10k steps on `spectrecipher/dataset_new_calib_20261028_085359`.
4. **Roll out** the policy on the robot with `lerobot-rollout`.
5. **DAgger / HIL:** run the policy and take over with the leader arm when it fails. The corrections are recorded as a new dataset.
6. **Fine-tune** the policy on the HIL data and repeat from step 4 (round 1, then round 2, and so on).

## 3. Files in this folder

| File | Description |
|---|---|
| [README.md](README.md) | This document. |
| [Instruction.md](Instruction.md) | Step-by-step guide to create a dataset and train a model, with the LeLab GUI and with the terminal. |
| [create_dataset.sh](create_dataset.sh) | Records a new teleoperated dataset with `lerobot-record` (30 episodes of 30 s, reset 10 s, 30 fps, pushed to the Hub). Edit `--dataset.repo_id` and `--dataset.root` before running. |
| [train.sh](train.sh) | Fine-tunes an ACT checkpoint on the HIL round-2 dataset (`spectrecipher/HIL_fine_tune_datasetlerobot_r2`) for 10k steps with batch size 8. It saves a checkpoint every 1000 steps, with W&B and Hub push turned off. |
| [rollout.sh](rollout.sh) | Runs a trained policy on the follower arm (`--strategy.type=base`) for 30 s, with no teleop and no recording. Use it to evaluate a checkpoint. |
| [rollout_dagger.sh](rollout_dagger.sh) | First DAgger run. It starts from the Hub policy `spectrecipher/so101-act-demo-v2`, records corrections to `spectrecipher/rollout_so101_dagger`, and pushes to the Hub. Duration is 300 s. |
| [rollout_dagger_hil_r1.sh](rollout_dagger_hil_r1.sh) | HIL round 1. It uses the 10k-step ACT checkpoint and records to `rollout_hil_so101_act_10k` for 60 s. It deletes an empty dataset folder left by a failed run and resumes the dataset if it already has episodes. |
| [rollout_dagger_hil_r2.sh](rollout_dagger_hil_r2.sh) | HIL round 2. It uses the checkpoint fine-tuned on round-1 data and records to `rollout_hil_so101_act_hil_r2`. It has the same cleanup and resume logic. |
| [so101_arm_robot_config.json](so101_arm_robot_config.json) | LeLab robot config: leader and follower ports, calibration files, and the two cameras (`Twist` rotated 180°, `top`). |
| [so101_arm.json](so101_arm.json) | Calibration of the **follower** arm: motor IDs 1-6, homing offsets and range min/max for each joint. |
| [pretrained_model/](pretrained_model/) | The trained ACT policy in LeRobot format. |
| [.gitignore](.gitignore) | Ignores `.venv/`, caches and build artifacts. |
| `.venv/` | Local Python environment with LeRobot installed (not tracked by git). |

### `pretrained_model/` contents

| File | Description |
|---|---|
| `config.json` | ACT policy config. Inputs are a 6-D `observation.state` and two RGB cameras (`Twist`, `top`, 3x480x640). The output is a 6-D `action`. |
| `model.safetensors` | Policy weights (about 200 MB). |
| `train_config.json` | Full training config (dataset, steps, batch size and so on). |
| `policy_preprocessor.json` and `policy_preprocessor_step_3_normalizer_processor.safetensors` | Input normalization (stats saved from training). |
| `policy_postprocessor.json` and `policy_postprocessor_step_0_unnormalizer_processor.safetensors` | Converts normalized outputs back to real joint actions. |

## 4. Usage

All commands assume the repo root and the LeRobot virtual environment:

```bash
cd /data/lerobot
source .venv/bin/activate
```

**Evaluate a policy** (edit `--policy.path` in the script first):
```bash
bash rollout.sh
```

**Collect HIL / DAgger data.** Connect both arms and cameras, then run:
```bash
./rollout_dagger_hil_r1.sh   # round 1
./rollout_dagger_hil_r2.sh   # round 2
```
While it runs, the policy controls the follower. Take over with the leader arm to correct mistakes. The episodes are saved to the dataset at `--dataset.root` and pushed to the Hub, so you need `huggingface-cli login`.

**Fine-tune on the new data:**
```bash
bash train.sh
```
Then point `--policy.path` in the next rollout script at `<output_dir>/checkpoints/last/pretrained_model`.

## 5. Notes

- **Ports** (`/dev/ttyACM*` or `/dev/serial/by-id/...`) and **camera indices** (0 and 2) depend on your machine. The `by-id` paths stay stable across reboots, so prefer them. Check them before each run.
- **Calibration:** the scripts pass `--robot.calibration_dir=/data/lerobot` and `--robot.id=so101_arm_latest`. LeRobot looks for `<id>.json` in that folder. Re-calibrate if you change motors or arms, and re-collect data if the calibration changes a lot.
- **Calibration files:** the scripts expect `so101_arm_latest.json` (follower) and `so101_leader_latest.json` (leader) in `/data/lerobot`, but neither is in the folder now (only `so101_arm.json` is). Copy them back from the LeLab/LeRobot calibration cache (`~/.cache/huggingface/lerobot/calibration`) or re-calibrate.
- **Absolute paths** inside the scripts (`/home/apc/...`) point to the author's machine. Update them to match your setup.
- The camera names (`Twist`, `top`) must match the policy's input feature names. Do not rename them without retraining.
