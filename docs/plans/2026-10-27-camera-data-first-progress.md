# Execution ledger — plan: docs/plans/2026-10-27-camera-data-first.md

## Setup

The approved conversation plan is recorded in the adjacent plan file.
Pre-flight: checkpoint gate determines whether trainer implementation and the
controlled experiment are needed; execute it before modifying training code.
Existing workspace has model/log artifacts and an unusual Git index state;
do not reset, stage or commit unrelated user files. Read-only gate evaluation
does not require a worktree or branch mutation.

## Task 1: in progress

Matched256-start screen launched in
logs/camera_800_gate_20261027_170933 with checkpoint hashes and exact commands
saved in plan.json. Results are validated with workspace_audit.read_result.

Baseline verification: `.venv/bin/pytest -q` completed with175 passed and2
optional-dependency skips (cv2, leapp). Every screened checkpoint has finite
student tensors and exactly equal embedded teacher tensors; optimizer steps
are consistent with the resumed800-update run. No training code changed.

Task1: complete. All10 processes exited0, reset/command provenance validated,
all initial poses matched exactly, hashes unchanged. No candidate qualified;
2200 is the strongest nominal-eligible baseline (229/256 nominal,80/256 wide).
Evidence: gate REPORT.md and selection.json. No1024 confirmation triggered.

## Task 2: in progress

Contract tests first: pytest failed on missing place_box.distillation as expected.
Ruling: implement on branch camera-data-first in the existing runnable workspace
without committing user changes — the current dependencies, models and previous
uncommitted fixes are required for validation; a clean worktree would omit them.
Cost if wrong: reduced filesystem isolation; preserve all unrelated files and
review only the explicit changed-file list. No Git reset, staging or commits.

Task2 implementation and paired-comparison support: new trainer21 contract
tests and comparison4 contract tests passed after observed missing-module RED
runs. Ruff check and format passed. Full regression suite:200 passed,2 skipped
(cv2/leapp),58.49s. Comparison CLI also consumed real gate audit files, validated
provenance/poses, reproduced8.984375-point wide gain and correctly declined
promotion. Its evidence is gate/comparison_cli/comparison.json.

GPU smoke:8 envs,2 new updates then2 resumed. Relative counters2→4,
optimizer77440→77696, replay512→1024, normalizer counts exactly256/update,
teacher tensors unchanged and student tensors finite. Standard default-agent
student-only playback loaded the new checkpoint and completed8 first episodes
(0 successes; this is a compatibility smoke, not an improvement evaluation).
Evidence: logs/camera_data_first_smoke_20261027/verification.json and play.json.
The additional32-update8-env smoke completed. At relative update36 the phase
rings held[4096,1786,1426,395]; all phases were present and capacity was bounded.
There were28 completed teacher episodes/28 teacher physical successes, zero
student episodes, teacher fraction1. Teacher remained exact, student finite,
normalizer increment9216, Adam81792. This is collection evidence, not student
accuracy. Evidence: phase_verification.json and REPORT.md in the smoke folder.
No400-update experimental arm launched.

Tasks3/4 preparation: direct user-launch commands, block endpoints, screen and
final audit commands, plus executable paired comparison are documented in
docs/CAMERA_DATA_FIRST.md. Actual400-update arms and final1024-start evaluations
remain dependent on user-launched training; no improvement claim is made.

## Final independent review and fix pass

Fresh reviewer found no Critical/Minor findings and one Important finding:
comparison could mix nominal/wide audits from different checkpoints. Regraded
Important because it could promote a nonexistent composite model. Three focused
tests reproduced the defect (all failed DID NOT RAISE), then all7 comparison
tests passed after retaining/validating checkpoint hashes and checking each job
against its manifest checkpoint. Final comparison reports the hashes. Real gate
audits also passed the corrected CLI: comparison_cli/comparison_verified.json.

Final: Ruling: long-run improvement/adoption remains pending user-run400-update
arms — explicitly agreed execution split — cost: student improvement is not yet
measured and no new trainer is adopted on the smoke evidence.
Final: Ruling: unrelated existing workspace changes/model archives are excluded
from this scoped review — preserve user work — cost: pre-existing issues outside
the changed-file set have not been independently audited here.

Documented Bash fences passed bash -n; both real Hydra agent configurations
resolved with identical student/teacher architecture, reset parameters, full
initial episodes, learning rate0.0001 and128 optimizer steps per update.
Evidence: logs/camera_data_first_smoke_20261027/command_verification.json.
No merge, push or publication was requested; keep the local implementation
branch and files available for the user's training runs. Retain this ledger
because no implementation commits were made; do not delete its evidence.

Final: fixed mixed-checkpoint comparison — three regression cases observed
RED→GREEN; final full suite203 passed,2 skipped in58.72s; Ruff passed. No
Critical/Important findings remain; no deferred minors. Task2 implementation
complete. Tasks3/4 tooling and commands ready; long training/final adoption
remain user-launched as agreed. Handoff/model index/current guide updated.

## User-requested commit snapshot

The user subsequently requested committing all work plus a summary/next action.
This supersedes the earlier decision to leave changes uncommitted. Include all
completed source/docs/tests and new raw log/checkpoint artifacts, preserving LFS
attributes; exclude the existing ignored runtime/caches. Refresh root inventories.
Retain this ledger as permanent project provenance. Fresh pre-commit pytest:
203 passed,2 optional skips in58.12s. Summary: docs/SUMMARY_NEXT_ACTION.md.
No push, merge or long training run is requested by this commit instruction.
