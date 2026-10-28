# Protected reference camera trainer

The50-update screen subsequently completed and failed:70.70% nominal and7.81%
wide versus2200 at89.45% and31.25%. Retain2200 and stop; no extension is queued.
All protected examples stayed unchanged and their fitting improved, but task
success regressed. [Results](../logs/camera_reference_20261028/screen50/REPORT.md).
Commands below describe the tested method; do not rerun it automatically.

The original controlled experiment regressed; camera2200 remains selected.
This opt-in trainer addresses demonstrated replay turnover by keeping earlier
successful teacher examples fixed. Retaining examples does not guarantee task
success or establish that forgetting was the only cause of regression.

## Behavior

`--agent rsl_rl_anchored_cfg_entry_point` selects the new trainer. Default box,
vial and original data-first entrypoints remain available.

Each128-row minibatch contains64 fresh,32 rolling phase replay and32 protected
reference examples. References are equally split between nominal/wide, with
four examples per phase per population. On the first update only, rolling
replay is empty:96 fresh and32 references. Both sources are sampled using
checkpointed private RNGs. There are still128 optimizer steps per update.

Only new rollout observations update normalization. Reference RGB is CPU FP16,
proprioception and teacher labels FP32. RGB noise outside[0,1] is retained;
labels/executed actions are bounded to[-1,1]. References never receive new
training observations. Their tensors stay unchanged while sampling RNGs advance.
Logged reference_samples, rolling_samples and fresh_samples distinguish sources.
Inherited replay_samples and replay_phase_* count both retained sources.

The assistance schedule, crop, CNN, teacher5381, physics, success criteria and
student-only inference are unchanged. Missing references, missing phases or
populations, invalid values and teacher identity mismatches fail before training.

## Prepare successful references

Run from the repository's existing Linux/WSL runtime. No new shell file is needed.
The prepared bank at `logs/camera_reference_20261028/reference.pt` uses32
successful nominal and32 successful wide teacher episodes, seed43. These are
training references, not holdout acceptance episodes.

To reproduce with new output paths:

```bash
BASE='checkpoints/2200.pt'
.venv/bin/python -m isaaclab_tutorial.tasks.place_box.reference_collection --checkpoint "$BASE" --population nominal --episodes 32 --num-envs 8 --seed 43 --output logs/new_reference_nominal.pt --device cuda:0 --visualizer none
.venv/bin/python -m isaaclab_tutorial.tasks.place_box.reference_collection --checkpoint "$BASE" --population wide --episodes 32 --num-envs 8 --seed 43 --output logs/new_reference_wide.pt --device cuda:0 --visualizer none
.venv/bin/python -m isaaclab_tutorial.tasks.place_box.reference_collection --nominal logs/new_reference_nominal.pt --wide logs/new_reference_wide.pt --capacity-per-phase 1024 --output logs/new_reference.pt
```

Only the first episode per environment counts in each batch. Snapshots are
copied before action every4 control steps and kept only when the episode meets
the existing physical success contract. The bank takes a randomized bounded
sample per phase/population, rather than the last rows in collection order.
It includes teacher tensor fingerprint, successful episode counts, checkpoint
hashes, reset parameters and starting poses. All outputs must be new files.

The prepared bank contains nominal phase counts[662,227,255,108] and wide
[1024,612,279,110],3277 examples total. Phase balancing samples with replacement;
small late-phase pools limit diversity. Collect more successful episodes before
assuming the bank covers the workspace comprehensively. Maximum bank size is
8192 examples with1024/phase/population, roughly300MB of RGB. Each training
checkpoint embeds this in addition to rolling replay (up to roughly600MB).

## Initialization and resume

An example bounded50-update screening block, starting from2200 and its optimizer:

```bash
.venv/bin/isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101-Camera --agent rsl_rl_anchored_cfg_entry_point --num_envs 128 --seed 43 --device cuda:0 --max_iterations 50 --resume_optimizer_state --checkpoint "$BASE" --run_name protected_reference_screen --visualizer none presets=newton_mjwarp,newton_renderer agent.algorithm.reference_path=logs/camera_reference_20261028/reference.pt agent.algorithm.learning_rate=0.0001 agent.init_at_random_ep_len=false env.events.reset_home.params.position_noise=0.1 env.events.reset_home.params.yaw_noise=3.141592653589793 env.events.reset_home.params.nominal_fraction=0.2 env.events.reset_home.params.sampling_seed=43
```

The50-update endpoint label is2249. This block completed under the user's
subsequent instruction and was not adopted. Evaluate256 nominal seed42 and256
wide seed45 starts using the default camera play agent, as in CAMERA_DATA_FIRST.md.
Compare with2200 and stop to inspect any nominal loss above3 percentage points.
Do not adopt on training loss, reference error or teacher-assisted success.

Resume with the anchored entrypoint, the anchored checkpoint and
`--resume_optimizer_state`. The checkpoint embeds references, rolling replay,
relative update and RNGs, so the original bank file is unnecessary. Embedded
references take priority even if reference_path names a missing file. A legacy
checkpoint without embedded references requires an explicit valid reference_path.
Legacy data-first imports preserve their assistance counter and rolling replay;
ordinary2200 imports begin relative update0 with empty rolling replay.

GPU collection and the short initialization/resume smoke validate mechanics,
not improved student task performance. Evidence is under
`logs/camera_reference_20261028/`. Default student-only playback completed8
episodes with0 successes; this compatibility smoke is not a trained-policy gain.
[Verification report](../logs/camera_reference_20261028/REPORT.md).
