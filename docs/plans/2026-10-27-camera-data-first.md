# Camera distillation improvement plan

Approved by the user before implementation. Keep the current wrist crop, CNN,
teacher 5381, physics and physical success criteria. No overhead camera, vial
changes, installed-library edits or per-run shell scripts. Long training runs
will be launched by the user using direct commands.

## 1. Completed 800-update run gate

Screen checkpoints 2000, 2200, 2400 and 2597 from the seed43 800-update run
against camera1798 on 256 nominal starts (seed42) and 256 full-workspace starts
(seed45). Rank by wide success after excluding candidates whose nominal success
falls more than 3 percentage points below1798; break ties by nominal success.
Confirm any apparent qualifying gain on 1024 nominal starts (seed51) and 1024
wide starts (seed48), with matched poses and student-only inference.

If confirmed wide improvement is at least10 percentage points and nominal drop
is at most3 points, retain that candidate and stop: skip the new trainer.
Otherwise use the strongest eligible checkpoint as the common baseline below.

## 2. Opt-in data-first box trainer

Implement place_box/distillation.py, BoxDataFirstDistillationRunnerCfg and the
camera-only rsl_rl_data_first_cfg_entry_point. Preserve the default trainer.

Assign one controller per episode, stable until done. Teacher probabilities by
relative experiment update: 0–49=1, 50–149=.5, 150–249=.25, 250–349=.1,
350–399=0. Teacher labels every visited state; clip labels and executed actions
to[-1,1]. Inference is always student-only.

Persist16384 transitions in four4096-capacity phase rings: approach,
grasp-before-lift, lift-before-placement, placed. Capture phase before action
from BoxProgress, aligned with copied observations and teacher labels. CPU
float16 RGB (preserve photometric noise outside[0,1]); float32 proprio/labels.
Each minibatch128 uses64 fresh and64 balanced replay samples, redistributing
empty buckets; fresh-only when replay is empty. Insert the rollout after its
optimization. Keep128 optimizer steps/update and normalize fresh data only.

Save versioned replay, counters, cursors and private RNG states with existing
model/optimizer keys. Legacy loads start empty replay at relative update0.
Resume restores algorithm state, assigning new controllers for reset envs.
Log separate student/teacher completed episode successes/counts, teacher control
fraction, phase counts, fresh/replay losses and occupancy.

Tests cover controller lifetime/dones/clipping, pre-action phase alignment and
copies, balanced bounded replay/fallback/FP16 tolerance, frozen teacher and
student observation isolation, checkpoint/RNG/schedule persistence, legacy
loads and student-only inference. Run8-env GPU smoke plus save/resume, pytest
and lint. Preserve all original checkpoints and raw evidence.

## 3. User-launched controlled training

Old trainer control versus new trainer, same baseline and optimizer state;
400 updates each in two200-update blocks. Both:128 envs, LR.0001,32 steps,
4 epochs,20% nominal/80% full workspace, gradient norm1, seed43 and explicit
reset sampling_seed43, full30s initial episodes (init_at_random_ep_len=False).
Private controller/replay RNGs. Save every100 and block ends; screen0/200/400
on256 nominal42/wide45. Provide direct WSL commands, no generated .sh files.

## 4. Final comparison

Best checkpoint per arm and common baseline:1024 nominal seed52 and1024 wide
seed49; student-only, matched initial poses. Report counts, paired outcomes and
descriptive paired bootstrap95% intervals (10000 resamples, seed0).
Adopt only if wide gain≥10 points and nominal drop≤3 points versus baseline.
Prefer the unchanged trainer if both qualify and the new trainer has no clearly
resolved wide advantage. Otherwise retain baseline and report unsuccessful or
inconclusive; no automatic training extension. These are improvement gates,
not deployment acceptance or changes to workspace_report thresholds.
