# Lower-rate camera continuation: first 200 updates

The run completed200 updates at learning rate0.0001 with a fresh optimizer. All logged scalars were finite, saved teacher weights matched5381, and the original normalization history was retained. See training_summary.json and resume_diagnostics.json.

| Checkpoint | Nominal | Wide workspace |
| --- | ---: | ---: |
| 1599 | 171/256 (66.80%) | 54/256 (21.09%) |
| 1700 | 0/256 (0.00%) | 23/256 (8.98%) |
| 1798 | 231/256 (90.23%) | 54/256 (21.09%) |

All starting poses match the1599 baseline exactly. These are small checkpoint-selection screens, not final acceptance.

Selected: **1798**. It tied1599 on wide success and improved nominal success by60/256. Midpoint1700 regressed badly before the final checkpoint recovered. No wide-workspace improvement has been demonstrated. The underlying cause of the transient regression remains unresolved.

## Remaining 200-update block

Continue1798 with the same0.0001 learning rate and preserve its optimizer state. The optimizer was already reset for block1; do not reset it again for block2. Keep teacher5381,128 environments, camera/physics settings and the same reset mixture. No second-block training was launched.

Run directly in PowerShell:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/e/issaclab_robotics && uv run --locked isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101-Camera --num_envs 128 --max_iterations 200 --seed 42 --run_name so101-box-workspace-camera-1798-lr1e-4-block2 --device cuda:0 --checkpoint logs/rsl_rl/so101_box_camera_distillation/2026-10-27_14-16-20_so101-box-workspace-camera-1599-lr1e-4-block1/model_1798.pt --resume_optimizer_state --visualizer none presets=newton_mjwarp,newton_renderer agent.algorithm.learning_rate=0.0001 agent.save_interval=100 env.events.reset_home.params.position_noise=0.1 env.events.reset_home.params.yaw_noise=3.141592653589793 env.events.reset_home.params.nominal_fraction=0.2'
```

Evaluate the next midpoint/final checkpoints against this same screen before selecting one. Edge-case trajectory inspection remains deferred.
