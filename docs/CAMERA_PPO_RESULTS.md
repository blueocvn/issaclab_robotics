# Latest Camera PPO results — 2026-10-28

All training and evaluation jobs are stopped. The latest run completed 500 additional
PPO iterations with a 60-second episode limit,starting from the 1000-iteration
continuation. These are 256-start selection screens,not deployment acceptance.

| 60-second evaluation|Nominal|Wide|
|---|---:|---:|
|Before extra 500 iterations:model 1048|253/256 (98.83%)|101/256 (39.45%)|
|After extra 500 iterations:model 1547|254/256 (99.22%)|101/256 (39.45%)|

The extra training produced no measured Wide gain. Paired 95% descriptive gain
intervals:Nominal[-1.17,+1.95] percentage points;Wide[-5.47,+5.47]. Both policies
used identical 60-second limits,reset protocols and initial poses. Checkpoint
hashes match across nominal/Wide audits and the on-disk files. No 1024-start
confirmation or hardware deployment was performed.

At 30 seconds,the earlier model 1048 scored255/256 Nominal and94/256 Wide.
Those scores must not be treated as a matched baseline for 60-second results.
The interrupted earlier-save comparison completed model 700 at256/256 Nominal,
78/256 Wide;checkpoint 100 evaluation was stopped on user request.

Training verification: 500 new iterations,labels 1048–1547;Adam 16800->24800;
actor normalization 17625088->21721088;critic 8601600->12697600. Actor and critic
are finite. LR 1e-5,fixed schedule,128 environments,64 rollout steps,4 epochs,
4 minibatches,seed67,20% nominal/80% full Wide resets. Camera,CNN,rewards and
physical-success requirements were unchanged. Only the per-run duration changed
to 60 seconds;the task default remains 30 seconds.

## Published models

- Distillation reference:[2200.pt](../checkpoints/2200.pt)
- 30-second Camera PPO:[model 1048](../checkpoints/camera_ppo/30s_1000/model_1048.pt)
- Latest 60-second Camera PPO:[model 1547](../checkpoints/camera_ppo/60s_500/model_1547.pt)

Use the native Camera PPO agent for PPO checkpoints. Example 4-env Wide playback:

```bash
.venv/bin/isaaclab play --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101-Camera --agent rsl_rl_camera_ppo_cfg_entry_point --checkpoint checkpoints/camera_ppo/60s_500/model_1547.pt --num_envs 4 --device cuda:0 --visualizer newton presets=newton_mjwarp,newton_renderer env.episode_length_s=60.0 env.events.reset_home.params.position_noise=0.1 env.events.reset_home.params.yaw_noise=3.141592653589793 env.events.reset_home.params.nominal_fraction=0.0
```

## Evidence and validation

Raw results:[60-second comparison](../logs/camera_ppo_60s_20261028/comparison.json),
[verification](../logs/camera_ppo_60s_20261028/verification.json),
[before Wide](../logs/camera_ppo_60s_20261028/before_wide/summary.json),
[after Wide](../logs/camera_ppo_60s_20261028/after_wide/summary.json),
[after Nominal](../logs/camera_ppo_60s_20261028/after_nominal/summary.json).
Historical raw files retain their original paths as provenance;resolve moved
models through[checkpoint_relocations.json](../checkpoint_relocations.json).

Full tests:227 passed,2 optional dependency skips (cv2/leapp);35 upstream
TorchScript warnings. Changed source/tests passed Ruff. Independent review
verified native PPO transfer/config migration and the optional audit horizon.
No further training is queued. Further changes require a new user instruction.
