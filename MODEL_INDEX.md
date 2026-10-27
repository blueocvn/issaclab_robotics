# Trained model archive

Model binaries in this repository use Git LFS. Run `git lfs install` and
`git lfs pull` after cloning. Preserve the `logs/` paths below: the training
commands and saved run configs refer to this layout.

| Role | Checkpoint | Validation status |
| --- | --- | --- |
| Latest expanded-workspace PPO refinement | [model_5381.pt](logs/rsl_rl/so101_box_state/2026-10-27_03-19-50_so101-box-workspace-F-refine/model_5381.pt) | 400-update fixed-rate refinement completed; independent acceptance pending |
| Best observed overnight expanded-workspace teacher | [model_4982.pt](logs/rsl_rl/so101_box_state/2026-10-26_19-41-12_overnight_20261027_ppo_camera_ppo_F_block6_try0/model_4982.pt) | 1,009/1,024 on selection seed43; 1,013/1,024 on unused seed45; still below full workspace gates |
| Accepted nominal box state teacher | [model_2196.pt](logs/rsl_rl/so101_box_state/2026-10-26_14-59-07_so101-box-release-v3-refine/model_2196.pt) | 3,072/3,072 nominal episodes across three seeds; not broad-workspace acceptance |
| Nominal box camera student | [model_1599.pt](logs/rsl_rl/so101_box_camera_distillation/2026-10-26_15-21-58_so101-box-camera-student/model_1599.pt) | 2,038/2,048 over two matching-start nominal audits; residual failures; not an expanded-workspace student |

All saved training checkpoints are included, together with available exported
policies, per-run agent/environment configs, TensorBoard events and evaluation
artifacts. Historical vial models and integration-smoke models are retained;
a smoke checkpoint is not evidence of learned task acceptance.

[checkpoint_manifest.json](checkpoint_manifest.json) lists every `.pt` artifact
with its relative path, size, SHA256 and whether it is a training checkpoint
or an exported policy. [publication_manifest.json](publication_manifest.json)
records the source snapshot and copied-file hashes.

For the current training workflow, see
[HANDOFF.md](docs/HANDOFF.md) and
[BOX_WORKSPACE_TRAINING.md](docs/BOX_WORKSPACE_TRAINING.md). The latest completed
refinement is not yet approved for camera distillation or real-robot deployment.
No expanded-workspace camera distillation has been completed. The camera
sim-to-real preset varies solid colors and image/camera parameters; printed
logos and texture variation are not implemented in the current task.

Example playback of the latest state refinement from the repository root:

```bash
uv run --locked isaaclab play --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101 --num_envs 1 --device cuda:0 --checkpoint logs/rsl_rl/so101_box_state/2026-10-27_03-19-50_so101-box-workspace-F-refine/model_5381.pt --visualizer newton presets=newton_mjwarp env.events.reset_home.params.position_noise=0.1 env.events.reset_home.params.yaw_noise=3.141592653589793
```

For the archived nominal camera checkpoint, use the camera task, its checkpoint
path from the table and `presets=newton_mjwarp,newton_renderer`. Hardware use
requires the separate calibration and transfer validation in
[SIM2REAL.md](docs/SIM2REAL.md).
