# Work summary and next action

Updated 2026-10-27. This snapshot includes the implementation, saved training
checkpoints, run configurations and evaluation evidence. The work is on local
branch `camera-data-first`.

## Completed work

- Installed and verified the WSL Isaac Lab runtime, dependencies and GPU task
  smoke tests. Historical models remain intact and use Git LFS.
- Fixed random-audit supplement start-index provenance and added a regression
  test. Recovered and validated the affected teacher audit.
- Evaluated teacher5381: grid **2,891/3,000 (96.37%)**, random
  **3,282/3,328 (98.62%)**, nominal **3,072/3,072**, historical regression **3/3**.
  The user accepted these results for camera distillation; they do not meet
  every original strict workspace acceptance threshold.
- Trained and evaluated the workspace camera student, screened lower-rate
  continuations, diagnosed phase coverage and tested frozen-policy camera
  framing. Keep the existing wrist crop. The full-frame resize reduced this
  frozen policy's success; retraining on that framing was not tested.
- Screened the completed 800-update seed43 run. No candidate achieved the
  agreed 10-point wide gain while limiting nominal loss to 3 points.
- Implemented the opt-in data-first trainer: episode-level teacher assistance,
  persistent phase-balanced replay, fresh-only normalization, bounded actions
  and labels, checkpointed replay/private RNGs, separate student/teacher metrics
  and student-only inference. Existing default box and vial trainers remain
  available.
- Added paired final comparison with matched poses/protocols, checkpoint
  identity checks, descriptive bootstrap intervals and the agreed adoption
  rules. Independent review's mixed-checkpoint finding was reproduced and fixed.

## Current camera baseline

These are matched **256-start selection screens**, not full acceptance:

| Checkpoint | Nominal successes | Wide successes |
| --- | --- | --- |
| 1798 reference | 225/256 (87.89%) | 57/256 (22.27%) |
| 2000 | 3/256 (1.17%) | 27/256 (10.55%) |
| **2200 experimental baseline** | **229/256 (89.45%)** | **80/256 (31.25%)** |
| 2400 | 34/256 (13.28%) | 62/256 (24.22%) |
| 2597 | 98/256 (38.28%) | 48/256 (18.75%) |

Use this exact baseline for both experimental arms:

```text
logs/rsl_rl/so101_box_camera_distillation/2026-10-27_16-26-29_so101-box-workspace-camera-1798-lr1e-4-seed43/model_2200.pt
SHA256: 3f5ad0ea7dabe634c40b1c0c82d4eee8a88b3ff8624aa88ce1801686a0c817b6
```

Teacher remains5381. No overhead camera, CNN change or physical-success
relaxation is part of this experiment. Behavior loss around0.12 is not an
accuracy threshold.

## Verification

- Fresh pre-commit regression run: **203 passed, 2 skipped** (optional cv2/leapp).
- Changed implementation/tests pass Ruff.
- Eight-env GPU smoke verified legacy initialization, save/resume, optimizer
  steps, fresh-only normalizer counts, all four replay phases, bounded rings,
  unchanged teacher tensors and standard student-only playback compatibility.
- The full-horizon smoke recorded28 successful **teacher-controlled** episodes.
  This is collection evidence, not student accuracy or an improvement claim.
- The documented Bash commands passed syntax checks; both real agent configs
  resolved to matching experiment settings.

## Next action

**Run the controlled experiment yourself, starting with the first200-update
block for each arm.** Use the setup and direct commands in
[CAMERA_DATA_FIRST.md](CAMERA_DATA_FIRST.md). No new per-run shell file is
needed. The 400-update experimental arms have not been launched.

1. Start unchanged-trainer control and data-first training independently from
   the exact2200 checkpoint **and its optimizer state**. Both use128 envs,
   LR0.0001, seed/reset seed43, full initial episodes, current wrist crop and
   the same20% nominal/80% full-workspace reset mixture.
2. After200 new updates, screen each `model_2399.pt` on256 nominal starts
   (seed42) and256 wide starts (seed45).
3. Resume each arm for its remaining200 updates. Endpoints are
   `model_2598.pt`; the new trainer's relative update counter reaches400.
4. Screen both400-update endpoints, then select the best nominal-eligible
   endpoint per arm. Confirm against the common baseline on1024 nominal
   starts (seed52) and1024 wide starts (seed49), all student-only.
5. Use the comparison command in the guide. Adopt only with a wide gain of
   at least10 percentage points and nominal drop of at most3 points. If both
   qualify, prefer control unless data-first has a resolved paired wide
   advantage. If neither qualifies, retain baseline; do not automatically
   extend training.

During training read student success **with completed student episode counts**;
teacher-assisted mixed success and behavior loss do not replace independent
student-only audits.

Detailed provenance: [HANDOFF.md](HANDOFF.md),
[checkpoint screen](../logs/camera_800_gate_20261027_170933/REPORT.md),
[GPU smoke](../logs/camera_data_first_smoke_20261027/REPORT.md),
[execution ledger](plans/2026-10-27-camera-data-first-progress.md).
