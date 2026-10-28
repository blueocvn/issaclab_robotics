# Camera PPO trial from2200 — completed2026-10-28

**Decision:retain2200.** Camera PPO completed50 new iterations and preserved
Nominal within the3-point limit, but its Wide gain did not meet the10-point
milestone. No automatic extension or1024-start confirmation is eligible.

| Native Camera PPO checkpoint | Wide | Nominal |
| --- | ---: | ---: |
| Before training:actor transferred from2200 | 83/256 (32.42%) | 236/256 (92.19%) |
| After50 PPO iterations | 87/256 (33.98%) | 230/256 (89.84%) |
| Change | +1.56 percentage points | -2.34 percentage points |

Wide paired descriptive95% interval:[-3.13,+6.64] points; Nominal:[-6.25,+1.56].
Four additional Wide successes do not establish a reliable improvement. This
bounded trial does not establish whether substantially longer PPO training would
help. It does establish correct initialized Camera PPO execution and a first
independent success screen. Do not select on training reward alone.

## Implementation

Added an opt-in BoxCameraPPORunnerCfg and rsl_rl_camera_ppo_cfg_entry_point;
default camera distillation and all vial/state agents stay available. Added
camera_ppo.py for strict full student-to-actor transfer and bounded native PPO
training. Added optional --agent support to workspace_audit, including random
supplements, without altering default audit commands or acceptance rules.

Actor inputs:wrist_rgb and proprioception. Critic input:privileged critic state.
This trial uses the standard RSL-RL PPO objective and environment rewards; no
teacher action labels, teacher action selection or distillation loss are used.
The source checkpoint supplies actor tensors only. PPO critic and Adam are fresh.

The first integration launch stopped before any update because raw configuration
serialization bypassed installed-version migration. A native runner-construction
regression reproduced the missing integration boundary. The final implementation
uses Isaac Lab's version migration, runner factory and canonical serialization.
Failed-launch settings/log are retained separately; trial_v2 is the only trained
PPO candidate.

## Actual execution

Source2200 SHA256:3f5ad0ea7dabe634c40b1c0c82d4eee8a88b3ff8624aa88ce1801686a0c817b6.
Every initial actor tensor equals the source student tensor, including normalizer
and Gaussian parameters. Measured pre-update deterministic action error on actual
simulator observations is exactly0.0. Critic unchanged by transfer; Adam empty.
Native initial checkpoint is trial_v2/initial.pt, separate from model0 saves.

128 environments,seed/reset67,20% nominal80% full Wide resets, full initial
episodes,64 control steps/iteration. Fixed LR1e-5,PPO clip0.1,entropy coefficient0,
4epochs*4minibatches=16 optimizer steps/iteration. Exploration starts from the
saved2200 distribution (~0.05 std), bounded0.01–0.15. No camera/CNN/reward/physics
or physical-success changes. Observation corruption remains standard training-on,
playback-off; both initial and endpoint use the same standard playback settings.

50 iterations completed:800 optimizer steps, finite actor and critic. Standard
fresh-rollout normalization advanced actor count9023488->9433088, exactly409600
transitions. The endpoint is trial_v2/training/model_49.pt (new PPO iterations
0–49, not old distillation iteration numbering).
Endpoint SHA256:89c47f8c837f54d5259a769695dc987b321fb753b155fca6cfa862853ebd749a.

## Audit integrity and historical reference

All four native initial/endpoint256-start jobs completed. Nominal seed42,Wide
seed45,explicit Camera PPO agent, matching reset parameters and exact paired
initial poses. Checkpoint identities match across each nominal/Wide pair and
on-disk files. comparison.json validates protocols and reports paired outcomes.

Historical2200 screens were229/256 Nominal and80/256 Wide. The fresh transferred
actor scored236 and83 despite identical tensors and matched recorded starting
poses. The comparison records this discrepancy instead of interpreting it as
learning. The native runner/agent path and a fresh execution differ from the
historical run; the cause of outcome variation has not been isolated. Therefore
use the fresh native initial policy as this PPO trial's reference, and treat
these small screens as selection evidence rather than precise deployment rates.
Audit insufficient_coverage status is expected at256 starts; all planned jobs
completed. No1024-start confirmation ran because the Wide gain was ineligible.

Stage counts before->after:Nominal grasp256->256,lift237->232,place236->231,
success236->230; Wide grasp174->163,lift99->97,place88->90,success83->87.
The Wide gain does not demonstrate an improvement in initial acquisition/lift.

## Verification and artifacts

- tests_final.log:222 passed,2 optional dependency skips (cv2/leapp),35 upstream
  TorchScript warnings. ruff.log:changed-file checks passed.
- REVIEW.md:independent review of transfer, buffer ownership, migration, native
  checkpoint/load semantics and optional agent propagation; no remaining blockers.
- trial_v2/initialization.json:exact actor identity/action parity,fresh critic/Adam.
- trial_v2/verification.json:actual PPO step count,finite models,normalizer count.
- comparison.json and compare.py:controlled paired validation and outcome intervals.
- initial_nominal/,initial_wide/,endpoint_nominal/,endpoint_wide/:raw audits.
- train_v2.log and trial_v2/training/:training metrics,native model files and events.
- ../../docs/CAMERA_PPO.md:reproduction,playback and audit commands. PPO checkpoints
  require --agent rsl_rl_camera_ppo_cfg_entry_point; do not load as distillation.

No original checkpoint replacement,hardware action,commit or push. No training
is running or queued. The PPO candidate remains an experimental artifact.
