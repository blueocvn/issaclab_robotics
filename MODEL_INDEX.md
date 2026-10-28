# Trained model archive

Model binaries in this repository use Git LFS. Run `git lfs install` and
`git lfs pull` after cloning. Published model files are consolidated under `checkpoints/`. Historical run configs retain original paths; use `checkpoint_relocations.json` to locate moved models.

| Role | Checkpoint | Validation status |
| --- | --- | --- |
| Latest Camera PPO,60-second episodes | [model1547](checkpoints/camera_ppo/60s_500/model_1547.pt) | Nominal254/256 (99.22%),Wide101/256 (39.45%);same-horizon source253/256 and101/256. No measured Wide gain. |
| Camera PPO,30-second episodes | [model1048](checkpoints/camera_ppo/30s_1000/model_1048.pt) | Nominal255/256 (99.61%),Wide94/256 (36.72%);selection screens only. |
| Common camera experimental baseline | [model_2200.pt](checkpoints/2200.pt) | Strongest nominal-eligible endpoint from the completed800-update run:229/256 nominal (89.45%),80/256 wide (31.25%); wide gain8.98 points vs fresh1798 missed the10-point improvement gate; selected for the controlled data-first experiment, not full acceptance; see [report](logs/camera_800_gate_20261027_170933/REPORT.md) |
| Latest expanded-workspace PPO refinement | [model_5381.pt](checkpoints/archive/so101_box_state/2026-10-27_03-19-50_so101-box-workspace-F-refine/model_5381.pt) | User accepted for camera distillation at observed grid2,891/3,000 and random3,282/3,328; below original strict gates; nominal3,072/3,072 and regression3/3 passed; see [report](logs/workspace_5381_acceptance_20261027_122310/REPORT.md) |
| Best observed overnight expanded-workspace teacher | [model_4982.pt](checkpoints/archive/so101_box_state/2026-10-26_19-41-12_overnight_20261027_ppo_camera_ppo_F_block6_try0/model_4982.pt) | 1,009/1,024 on selection seed43; 1,013/1,024 on unused seed45; still below full workspace gates |
| Accepted nominal box state teacher | [model_2196.pt](checkpoints/archive/so101_box_state/2026-10-26_14-59-07_so101-box-release-v3-refine/model_2196.pt) | 3,072/3,072 nominal episodes across three seeds; not broad-workspace acceptance |
| Nominal box camera student | [model_1599.pt](checkpoints/archive/so101_box_camera_distillation/2026-10-26_15-21-58_so101-box-camera-student/model_1599.pt) | 2,038/2,048 over two matching-start nominal audits; residual failures; not an expanded-workspace student |
| Expanded-workspace box camera student | [model_1599.pt](checkpoints/archive/so101_box_camera_distillation/2026-10-27_12-56-22_so101-box-workspace-camera-5381/model_1599.pt) | Completed1,600 iterations; exact frozen teacher5381; independent grid455/3,000 (15.17%), random673/3,328 (20.22%), nominal2,152/3,072 (70.05%), case5273/3; see [report](logs/camera_1599_evaluation_20261027_132245/REPORT.md) |
| Selected lower-rate workspace camera continuation | [model_1798.pt](checkpoints/archive/so101_box_camera_distillation/2026-10-27_14-16-20_so101-box-workspace-camera-1599-lr1e-4-block1/model_1798.pt) | Retained after the full400-update experiment; matched256-start screen: nominal231/256 (90.23%), wide54/256 (21.09%); later1900/1997 regressed; wide tied1599, nominal improved; full acceptance not established; see [report](logs/camera_lr1e4_block2_screen_20261027_151453/REPORT.md) |
| Block2 camera endpoint, not selected | [model_1997.pt](checkpoints/archive/so101_box_camera_distillation/2026-10-27_15-10-21_so101-box-workspace-camera-1798-lr1e-4-block2/model_1997.pt) | Completed the remaining200 updates with optimizer resumed at0.0001; nominal164/256 (64.06%), wide14/256 (5.47%); retain1798 |

The previous published archive and all Camera PPO saves are included, with published exports and experiment evidence. Unselected replay-heavy experiment checkpoints and large raw collection bundles remain local. No files were deleted. Historical vial models and integration-smoke models are retained;
a smoke checkpoint is not evidence of learned task acceptance.

[checkpoint_manifest.json](checkpoint_manifest.json) lists every `.pt` artifact
with its relative path, size, SHA256 and whether it is a training checkpoint
or an exported policy. [publication_manifest.json](publication_manifest.json)
records the source snapshot and copied-file hashes.

For the current training workflow, see
[HANDOFF.md](docs/HANDOFF.md) and
[BOX_WORKSPACE_TRAINING.md](docs/BOX_WORKSPACE_TRAINING.md). See [latest PPO results](docs/CAMERA_PPO_RESULTS.md) for the current experiment. Real-robot deployment is not authorized.
Expanded-workspace camera distillation completed1,600 iterations from5381;
its independent evaluation completed and remains below the teacher, including
nominal starts. The earlier five-iteration smoke model_4.pt
is only an initialization check. See `logs/camera_5381_smoke_verification.json`. The camera
sim-to-real preset varies solid colors and image/camera parameters; printed
logos and texture variation are not implemented in the current task.

Example playback of the latest state refinement from the repository root:

```bash
uv run --locked isaaclab play --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101 --num_envs 1 --device cuda:0 --checkpoint checkpoints/archive/so101_box_state/2026-10-27_03-19-50_so101-box-workspace-F-refine/model_5381.pt --visualizer newton presets=newton_mjwarp env.events.reset_home.params.position_noise=0.1 env.events.reset_home.params.yaw_noise=3.141592653589793
```

For the archived nominal camera checkpoint, use the camera task, its checkpoint
path from the table and `presets=newton_mjwarp,newton_renderer`. Hardware use
requires the separate calibration and transfer validation in
[SIM2REAL.md](docs/SIM2REAL.md).
