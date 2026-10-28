# Camera PPO fine-tuning from2200 — implementation and trial

**Goal:** Test reward-based camera-policy fine-tuning from the retained2200.
**Spec:** User chose2200 initialization and explicitly authorized trying Camera PPO.
**Architecture:** Add opt-in BoxCameraPPORunnerCfg and a native OnPolicyRunner
entrypoint. Strictly transfer student tensors to PPO actor; critic and PPO Adam
remain fresh. Native PPO checkpoints and standard playback use the opt-in agent.
Extend workspace_audit with an optional agent argument, preserving default jobs.
**Alternatives:** State-PPO continuation does not train vision; further imitation
already regressed. From-scratch Camera PPO discards2200 skills. Choose initialized
camera PPO as requested. No teacher imitation loss or teacher-controlled actions.
**Tech:** Existing Isaac Lab/RSL-RL/PyTorch, WSL GPU runtime.

## Constraints

Existing dirty camera-data-first workspace retained (no duplicate runtime or
reset/stage/commit/push); user waived further approval prompts. Preserve source
checkpoints, existing trainers, camera, physics and physical-success conditions.
Current task is separately authorized after previous distillation stop. No
hardware work. No blind1000-iteration extension or adoption by training reward.

## Bounded trial

128envs, seed/reset67, full initial episodes,20% nominal80%full Wide resets.
50 iterations,64 control steps/iteration,4epochs*4minibatches=16 optimizer
steps/iteration (800 total). LR1e-5 fixed, PPO clip0.1, entropy coefficient0,
actor exploration std inherited from2200 (~0.05), bounded std0.01–0.15.
Fresh privileged critic and fresh Adam; actor normalizer starts from2200 and
standard PPO updates normalization using fresh rollout observations.
Save initial nativePPO checkpoint separately, then train50 iterations once.
Audit initial and endpoint using256-env nominal42 and Wide45 protocols, opt-in
PPO agent for both. Require pre-update actor parity and matching initial audit
poses to historical2200. Select only if Wide gains>=10pp and Nominal loses<=3pp;
confirm on independent1024 starts if eligible. Otherwise retain2200 and stop.

## Tasks

- [x] Tests RED:config opt-in, strict complete student-to-actor transfer, preserved
  actor normalization, critic/optimizer untouched, invalid/nonfinite/mismatched
  tensors rejected before mutation, actual CNN prediction parity; audit optional
  agent preserves defaults and appears in all planned jobs.
- [x] Implement config/entrypoint, warm-start utility and bounded training CLI.
  Run targeted tests GREEN and relevant existing tests, Ruff changed code.
- [x] Independent review, then real GPU parity check/native initial checkpoint;
  train50 iterations, verify finite models and real PPO optimizer steps.
- [x] Initial+endpoint matched nominal/Wide screens; compare results; conditional
  independent confirmation only if eligible. Report and update handoff.

## Review focus

Actor must never receive teacher state; no distillation optimizer/critic copied;
normalization complete; initialization before any action update; all evaluation
jobs use native PPO agent; initial checkpoint cannot be overwritten by model0;
exact source identity and iteration/optimizer counts; evaluation success unchanged.

Execution ledger:initial7 tests RED (missing implementation),then37 targeted
GREEN. Initial GPU launch failed before any update because raw config bypassed
installed-version migration. Added native-runner regression RED then GREEN using
standard migration/factory and canonical serialization. Final suite222 passed,
2 optional skips; Ruff clean. Independent review found no remaining blockers.
trial_v2 action parity0.0, all actor tensors match2200, critic unchanged/fresh
Adam. Completed50 iterations/800 steps, finite actor+critic, normalization count
9023488->9433088. Four native initial/endpoint screens now running.


Completed:initial236 Nominal/83 Wide; endpoint230 Nominal/87 Wide,all256-start.
Matched commands/poses/checkpoint identities verified. Nominal-2.34pp,Wide+1.56pp.
No qualifying Wide gain; skip1024 confirmation,retain2200 and stop. Full results:
logs/camera_ppo_20261028/REPORT.md. No training running or queued.

