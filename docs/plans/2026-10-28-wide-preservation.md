# Wide baseline-action preservation experiment

**Goal:** Improve Wide while retaining2200's successful behavior.
**Spec/design:** User accepted expanding successful2200 examples followed by a
short constrained training test. Earlier permission waives further approvals.
**Approaches:** Another teacher-only replay run already failed; camera changes
require a separate adaptation experiment. Choose a standalone offline experiment
with explicit baseline-action preservation. This is a research spike, confined
to logs, not a new production trainer or proven fix.
**Architecture:** Reuse frozen camera2200+teacher5381, unchanged CNN/camera/task.
Collect128 fresh nominal and128 fresh Wide student-controlled first episodes,
seed63. Use only successful episodes for preservation, target=2200 actions.
Wide teacher demonstrations and baseline-visited Wide states from seed61 supply
teacher improvement targets. On improvement inputs also penalize deviation from
saved baseline predictions. Objective=MSE_teacher+4*MSE_preservation+MSE_deviation.
Preservation samples equally split nominal/Wide and balanced by available phase;
improvement samples equally split teacher/student trajectories and balanced by
available phase. Minibatch64 improvement+64 preservation. Seed67 sampling.
Restore original Adam state, set LR1e-5, freeze normalization statistics.

## Constraints

No physical-success relaxation, camera/CNN change, hardware action, model
replacement, commit or push. Existing dirty feature branch retained to reuse
runtime; all new scripts/data/models under logs/camera_wide_focus_20261028.
Source checkpoints immutable. Training/collection poses separate from audits.
No claim of recovery from intermediate states from labels alone.

## Steps

- [x] Expand collector arguments (seed, output directory, random128+128 poses),
  preserve previous48-pose behavior. Collect and verify successful phase coverage.
- [x] Implement offline experiment with explicit target sources, bounded actions,
  finite gradients, unchanged teacher/normalizer, correct Adam steps and model
  save/load. Store dataset hashes, sampler RNG and exact step count.
- [x] Train100 optimizer steps; screen256 nominal seed42 and256 Wide seed45 with
  existing workspace_audit. Check protocol/pose/hash pairing with2200.
- [x] Stop if Nominal loss>3pp OR Wide below baseline. Otherwise permit two more
  100-step blocks (max300), screening each. If candidate gains>=10pp Wide and
  loses<=3pp Nominal, confirm on independent1024 starts (seeds52/49) versus2200.
- [x] Report results and update handoff. Retain2200 unless confirmation qualifies.

## Verification and review focus

Check unsuccessful preservation exclusion, target-source distinction, phase and
population balancing, no audit-pose overlap, finite/bounded tensors, teacher and
normalizer immutability, inherited Adam counter increments, default student-only
playback compatibility. Offline losses are never selection metrics. Pilot
Nominal5/16 versus audit229/256 is a warning about small cohorts/protocol changes;
all candidate selection retains the exact256-env protocol.

## Execution ledger

Expanded seed63 collection completed:75/128 successful Nominal,24/128 successful
Wide. Train preservation63 Nominal+15 Wide; episode holdout12 Nominal+9 Wide.
First100 optimizer steps complete:Adam77184->77284, LR1e-5, unchanged teacher
and normalizer, finite gradients, heldout action MSE nominal0.002194,Wide0.003503.
These errors are not task-success claims. Independent review found no blocking
issues; added controller metadata and pose-file reuse guards. Student-only
256-start Nominal/Wide screens running.


Outcome:Nominal161/256,Wide79/256. Stop after100 steps; retain2200.
No continuation or1024-start confirmation. Found collection/playback observation
corruption mismatch:preservation only covered augmented collection inputs.
This is a coverage gap, not proof of causality. Future collection must explicitly
record corruption flags and cover clean inference inputs before another test.

