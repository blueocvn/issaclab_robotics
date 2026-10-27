# Camera checkpoint selection screen

Each candidate used256 identical nominal starts (seed42) and256 identical wide-workspace starts (seed45). This is selection evidence, not acceptance. All checkpoint hashes and exact initial-pose matches were verified.

| Checkpoint | Nominal | Wide workspace |
| --- | ---: | ---: |
| 1200 | 122/256 (47.66%) | 31/256 (12.11%) |
| 1300 | 121/256 (47.27%) | 17/256 (6.64%) |
| 1400 | 103/256 (40.23%) | 54/256 (21.09%) |
| 1599 | 171/256 (66.80%) | 54/256 (21.09%) |

Selected: **1599**. It tied1400 on wide successes and had substantially more nominal successes. The hypothesis that an earlier checkpoint would be stronger was not supported by this screen.

## Next controlled experiment

Continue1599 at learning rate0.0001 with a fresh optimizer. Keep frozen teacher5381, student weights, camera settings, physics,128 environments and the20% nominal/80% wider-workspace mixture unchanged. Split the approved400 updates into two200-update blocks, evaluating between them. No continuation training was launched.

Run the first block directly in PowerShell:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/e/issaclab_robotics && uv run --locked isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101-Camera --num_envs 128 --max_iterations 200 --seed 42 --run_name so101-box-workspace-camera-1599-lr1e-4-block1 --device cuda:0 --checkpoint logs/rsl_rl/so101_box_camera_distillation/2026-10-27_12-56-22_so101-box-workspace-camera-5381/model_1599.pt --no-resume_optimizer_state --visualizer none presets=newton_mjwarp,newton_renderer agent.algorithm.learning_rate=0.0001 agent.save_interval=100 env.events.reset_home.params.position_noise=0.1 env.events.reset_home.params.yaw_noise=3.141592653589793 env.events.reset_home.params.nominal_fraction=0.2'
```

After this block, evaluate the saved student on the same nominal/wide screen before deciding whether to run the remaining200 updates. The lower learning rate remains an experiment, not a confirmed fix.
