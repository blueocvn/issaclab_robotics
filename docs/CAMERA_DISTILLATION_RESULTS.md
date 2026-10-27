# Camera student results

The user completed 1,600 camera distillation updates with 128 environments in
`logs/rsl_rl/so101_box_camera_distillation/2026-10-26_15-21-58_so101-box-camera-student`.
The latest checkpoint is `model_1599.pt`. Its frozen state teacher is the refined
PPO `model_2196.pt`; inference uses the camera student.

All recorded training scalars are finite. Mean behavior loss decreased from
0.021947 over the first 50 updates to 0.002792 over the last 50 (87.3%).
The last 50 training success estimates averaged 99.80%.

## Exact episode audits

Each audit used 1,024 first episodes, seed42, the original success conditions,
and no scripted assistance or physics determinism flag.

| Checkpoint | Success | Timeouts | Failed environment IDs |
| --- | --- | --- | --- |
| Latest1599, first audit | 1,020/1,024 (99.6094%) | 4 | 55,177,238,885 |
| Latest1599, unchanged repeat | 1,018/1,024 (99.4141%) | 6 | 112,161,339,416,630,1008 |
| Earlier1500 | 1,022/1,024 (99.8047%) | 2 | 270,709 |

Every episode grasped and lifted; no audit recorded drops, out-of-bounds, or
unstable-robot terminations. In the first latest-checkpoint audit, all four
timeouts reached the target upright but failed to complete release/withdrawal.
The repeat also included four episodes without on-target/upright completion.

All 1,024 initial box poses match across these audits. Nevertheless, failures
changed on an unchanged repeat. An instrumented replay of the first four failed
starts also succeeded in all four cases. The source of rollout variation has
not been isolated; this does not erase the original failures or prove a fix.
The two latest-checkpoint audits total 2,038/2,048 successes (99.5117%), using
repeated starts, rather than independent held-out seeds. One audit of1500 is
insufficient to establish it as the better checkpoint.

The student exceeds the original 90% nominal acceptance threshold, but has
residual failures. The requested next action is visual playback. Further camera
validation should precede any hardware transfer; no additional long training
or hardware operation has been started by the agent.

## Play the latest camera checkpoint from PowerShell

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/work/robotics/IsaacLabTutorial; GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA uv run --locked isaaclab play --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101-Camera --num_envs 1 --device cuda:0 --checkpoint logs/rsl_rl/so101_box_camera_distillation/2026-10-26_15-21-58_so101-box-camera-student/model_1599.pt --visualizer newton presets=newton_mjwarp,newton_renderer'
```

Press Ctrl+C to stop. Single-environment playback is a visual check, not the
1,024-environment acceptance audit.

## Evidence

Repository-relative artifacts under `logs/`:

- `box_camera_training_summary.json`, `box_camera_training_series.json`, and
  `box_camera_training_curves.png` / `.svg`.
- `box_camera_1599_seed42_evaluation.json` and
  `box_camera_1599_seed42_repeat_evaluation.json`.
- `box_camera_1500_seed42_evaluation.json`.
- `box_camera_1599_failed_episodes_trace.json`.
- `box_camera_validation_summary.json`: recomputed episode counts, finite
  training scalars, matching starts, and instrumented replay outcomes.
