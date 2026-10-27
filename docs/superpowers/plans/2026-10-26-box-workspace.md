# Box Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Prepare opt-in wider-position/yaw PPO resets and reproducible workspace audits, verified on this PC, with long training left to the user.

**Architecture:** Keep the nominal reset path compatible. Add pure sampling/grid helpers and pass opt-in settings through the existing reset event. Extend exact evaluation with reset provenance; provide a subprocess audit runner and count-based reporting for fixed grids, random starts, nominal regression and saved case527.

**Tech Stack:** Existing Python3.12, PyTorch, Isaac Lab/Newton, RSL-RL and pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-26-box-workspace-design.md` (approved2026-10-26).

## Global Constraints

- +/-10 cm in both X and Y; upright yaw[-pi,pi]; fixed paper.
- Preserve existing success, rewards, nominal physics,54D state,6D actions and vial tasks.
- Training mixture20% nominal /80% current stage; audits use100% their stated distribution.
- User launches long training. Agent runs bounded tests and short integration checks only.
- Work in the current feature checkout: its box implementation, checkpoints and environment are uncommitted local state; a fresh worktree would omit them. Preserve unrelated changes and do not commit this existing work implicitly.

## Review Focus

- Partial reset changes only selected environment poses and provenance.
- Nominal default resets consume the original random sequence; explicit audit seeds are independent of model initialization.
- Batched grids include all boundary/yaw cases and repeats; random batches do not repeat the same seed stream.
- Audit scores cannot silently include nominal-mixture starts, missing cases or insufficient samples.
- Resuming a stage preserves checkpoint identity and uses the intended optimizer; short integration verifies finite losses before user commands.

### Task1: Optional reset mixture and controlled starts

**Files:** create `src/isaaclab_tutorial/tasks/place_box/workspace.py`; modify `mdp.py`, `env_cfg.py`; create `tests/test_box_workspace.py`.

**Interfaces:** `STAGES` maps A-F to(position half-width,yaw half-width). `sample_offsets(count, device, position_noise, yaw_noise, nominal_fraction=0, generator=None)` returns `(offsets[N,3], nominal_mask[N])`. `grid_offsets(indices, position_noise, yaw_noise)` maps global case indices to offsets, with200 cases per repeat. `reset_home` gains nominal_fraction, reset_mode(random/grid/replay), sampling_seed, grid_start_index and replay_path; defaults preserve existing behavior. Selected reset provenance is stored per environment.

- [x] Write tests for mixture endpoints/statistics, stage ranges, invalid finite/probability inputs, grid corners/8 orientations/repeats and normalized reset quaternions.
- [x] Run focused tests; observe missing behavior fail before implementation.
- [x] Implement helpers/reset integration, deterministic independent audit RNG and validated replay poses; record provenance without changing rewards.
- [x] Test actual reset with fake assets for partial reset isolation and saved starts; verify nominal RNG sequence compatibility.
- [x] Run focused tests and existing box tests; inspect output.

### Task2: Reproducible audits and acceptance reporting

**Files:** extend `evaluation.py`; create `workspace_audit.py`, `workspace_report.py`; extend `test_box_evaluation.py`; create `tests/test_box_workspace_audit.py`.

**Interfaces:** evaluation JSON includes reset parameters and per-episode distribution/case ID. `summarize(episodes, population, position_noise, yaw_noise, policy)` returns counts, stage flags, spatial/yaw bins, Wilson interval and independent acceptance status. Audit CLI accepts checkpoint, teacher/camera, stage, population(grid/random/nominal/regression), seed groups, batch size, output directory and dry-run. Calls the installed play CLI in isolated child processes. Grid index offsets and explicit sampling seeds make starts independent of actor initialization and preserve coverage across batches.

- [x] Write failing tests for provenance retained before auto-reset, immutable first-episode counting, complete batched200-case coverage, random seed derivation, CLI parameter validation, sample shortage and failure gates.
- [x] Implement audit extension and runner; grid default5 repeats and seeds45/46/47; random1024/group with256 supplements only for coverage; nominal1024/group42/43/44; regression saved case527 three times.
- [x] Implement separate report JSON per population, raw log preservation, checkpoint/command metadata and explicit failure/insufficient-coverage states; do not pool populations.
- [x] Add deterministic tests that nominal-mixture contamination and missing/duplicate grid cases cannot pass; test expected nominal teacher/student thresholds and no erased regression failures.
- [x] Run focused and full suites; inspect all failures.

### Task3: GPU verification, review and user runbook

**Files:** create `docs/BOX_WORKSPACE_TRAINING.md`; update task guide/progress/spec status and this plan.

- [x] Run CPU tests and Ruff on changed source/tests.
- [x] Run two PPO updates from teacher2196 using stage A mixture; verify finite training losses, saved checkpoint and reset provenance/ranges.
- [x] Run a bounded controlled-grid audit with teacher and camera (same cases) and inspect saved initial poses; no acceptance claims from a smoke-sized sample.
- [x] Verify runner dry-run commands for final populations and document stage A200-update command plus audit/promotion procedure. Do not launch the long command.
- [x] Request one independent whole-change review, fix material findings with failing regressions first, rerun appropriate checks.
- [x] Record evidence and remaining limits. Leave changes reviewable in place.

## Execution ledger

- Setup: approved design; native inline execution. Current feature branch `feat/so101-box-placement`, base `7efef771fa0bfbcb9683193ff100ce9f015fb17e`.
- Ruling: reuse current checkout rather than create a worktree or make incidental commits, because the required box feature and runtime are untracked local work; copying/moving them would add risk without isolating the effective starting state.
- Pre-flight: reset provenance produced in Task1 is captured before auto-reset in Task2; fixed-case indices remain global across audit batches. Task3 consumes the exact same CLI event settings, not a separate training implementation.
- Task1 focused gate:12 new tests failed on missing functionality before implementation;25 reset/config/evaluation tests now pass (`logs/workspace_task1_red.log`, `logs/workspace_task1_green.log`).
- Task1 complete: full suite includes all new reset tests. GPU smoke exposed int32 framework IDs; two failing regressions reproduced the mismatch, then explicit int64 provenance corrected it. Nominal RNG defaults remain tested.
- Task2 complete:11 initial failures reproduced missing audit/report functionality; subsequent seed/physical-start tests observed red then green. Final independent review found three accounting issues; five tests (including two GPU dtype cases) failed before fixes, then two batch-binding regressions failed before fixes. Full suite after fixes:164 passed,2 optional skips. Ruff clean.
- Task3 runtime: two-update512-env PPO smoke succeeded with finite scalar losses; source optimizer step88,000 versus fresh step80, reset batch105 nominal/407 stage starts with valid ranges. Dry-run budgets: grid3,000, random/stage/nominal3,072 each, regression3. Matched-grid checks pending final camera batch.
- Final review: one read-only reviewer; all three material findings fixed with regressions. No deferred minor findings.
- Final: Ruling: reviewer left GPU and optimizer verification to the parent — bounded runtime checks above cover these directly, with long training still user-launched — broader competency requires the planned training/audits, not these smoke results.
- Matched-grid integration complete:200 teacher/camera initial poses match within1e-7, including all boundary/yaw cases. Frozen teacher4/200 and camera6/200; both correctly report insufficient coverage for the incomplete final audit. All implementation checks passed.

### Task4: User-authorized overnight execution and shutdown

User now explicitly authorizes the agent to run PPO through camera distillation
and shut down the PC after completion or an unresolved failure. This overrides
earlier manual-launch and leave-on instructions. Automate existing commands and
gates without changing success criteria. Save a durable journal/status/report,
bound convergence attempts, retry recoverable execution/OOM failures, keep the
PC awake while running, and request graceful Windows shutdown after final logs
are saved. Never claim successful training on a failed or incomplete audit.

- [x] Add failing orchestration tests for stage gating, checkpoint selection,
  teacher-before-student acceptance, failure persistence and bounded retries.
- [x] Implement `workspace_pipeline.py` and a Windows hidden keep-awake/shutdown
  wrapper; test command generation without training or shutdown.
- [x] Independently review the new controller, verify scripts, then launch one
  durable run and confirm the first PPO block is actually training.

Task4 complete: user-authorized overnight controller launched at
`logs/overnight_20261027_ppo_camera`; native wrapper PID32052 and Linux controller
PID729 verified alive independently of the launch shell. Stage A worker PID778
is producing PPO updates. Final suite174 passed,2 optional skips; Ruff clean.
The ten controller tests include three recovery findings observed failing
before fixes, including a real contained process-tree cleanup test. Windows
helper syntax and native wake request verified. Checkpoint/task competence
remains pending the live pipeline's gates; launch is not training completion.

Ruling: explicit user permission now overrides manual-launch and leave-on
preferences; automatically power off after success or unresolved failure.
Ruling: bound unattended work to8 blocks/stage, three camera blocks and10 hours
overall; a nonconvergent run retains usable checkpoints and reports failure
rather than claiming success. The limits may stop a still-improving policy;
the saved checkpoints support a later continuation.
Final controller review: all three material recovery findings fixed; no
deferred minor findings. GPU/convergence is covered by actual running jobs and
gate reports, not assumed from unit tests. Actual shutdown is deliberately not
executed during verification; the authorized helper will request it after the
pipeline ends, with a recovery grace. Unsaved applications may block graceful
shutdown.
