# Protected reference replay implementation plan

> **For agentic workers:** Use superpowers:executing-plans task by task.

**Goal:** Preserve earlier successful teacher examples while learning new states.
**Architecture:** Add an opt-in anchored trainer, leaving previous trainers available.
Use64 fresh,32 rolling and32 fixed reference examples per128-row minibatch.
Reference data is phase-balanced and balanced between nominal/wide populations,
collected only from successful frozen-teacher episodes. It is never overwritten
by training, and travels in resumable checkpoints with private sampling RNGs.
**Tech stack:** Existing PyTorch/RSL-RL/Isaac Lab and pytest, WSL runtime.
**Spec:** The proposal approved in conversation: protected reference bank,
rolling replay,50/25/25 sampling, initialize from2200, independent success gates.

## Constraints and rulings

- Keep teacher5381, crop, CNN, physics and success contract unchanged.
- No extension of completed training during implementation; only short GPU smoke.
- User explicitly waived further approval prompts; execute inline.
- Work in existing camera-data-first feature branch and runnable workspace:
  preserve models/logs and prior documentation edits; no reset/stage/commit/push.
  Ruling: avoid duplicating the large runtime/model tree; isolation is reduced.
- Reference bank is explicitly prepared before training; missing/incompatible data
  fails closed. Resume embeds the bank so the original file is unnecessary.
- Existing data-first checkpoints without protected references need an explicit
  bank and preserve their relative assistance counter when imported.

## Review focus

Failed episode exclusion; missing phase/population; wrong teacher fingerprint;
reference mutation/aliasing; legacy checkpoint/resume without external files.

## Task1: reference construction and opt-in trainer

Files: new place_box/reference_replay.py, place_box/reference_collection.py,
tests/test_box_reference_replay.py; scoped changes distillation.py, agents.py,
__init__.py.

- [x] Write tests: reference excludes failed episodes; all phases/populations
  required; balanced sampling; training leaves reference tensors unchanged;
  fresh/rolling/reference counts; missing/wrong-teacher bank rejected;
  checkpoint roundtrip restores data/RNGs without file; legacy trainers unaffected.
- [x] Observe failure, implement, run focused tests and old distillation tests.
- [x] Add CLI that collects first teacher-controlled episodes for each population,
  samples every4 control steps, retains only physically successful episodes,
  writes bounded bank plus teacher fingerprint and collection provenance.

## Task2: integration verification and documentation

- [x] Collect a small real teacher bank on GPU, verify success provenance.
- [x] Run8-env two-update training and resume smoke from2200; verify64/32/32
  after first rollout, reference immutability, optimizer counts, frozen teacher,
  fresh-only normalization and student-only inference compatibility.
- [x] Run full pytest and changed-file Ruff; independent scoped code review.
- [x] Document collection, initialization, resume, storage cost and limitations.
  Keep2200 selected; no accuracy improvement claim from smoke evidence.

## Execution ledger

Pre-flight: collector produces versioned reference payload consumed by trainer;
teacher fingerprint covers exactly teacher_state_dict tensors in camera2200.
Legacy data-first behavior remains unchanged through an overridable sampling hook.

Task1 complete: observed missing-module RED, then28 focused tests passed;
collector test failed on missing module before implementation, then29 focused
tests passed. Added merge/provenance/registration and actual batch-source checks.
Baseline suite203 passed,2 skipped. No new runtime dependencies were installed.

Independent scoped review found one Important issue: eager reference_path reading
prevented resume from an embedded bank when the original path was gone. A new
test reproduced FileNotFoundError before the fix; constructor now defers external
reads.11 reference tests passed afterward. Review identified no other Critical,
Important or Minor findings. GPU behavior, accuracy and causal model conclusions
were declined by reviewer and verified separately or explicitly not claimed.

Task2 verification: final full suite214 passed,2 optional skips(cv2/leapp),55.67s;
changed-file Ruff check and format passed. GPU bank collection32/32 nominal and
32/32 wide successful teacher episodes. Bank SHA256
e79f69ff7754b42fd8dadd4f24e7b672dac45f43bcfaa1dcd7db059ecb983fcf.
Two-update smoke plus two-update resume reached relative update4, optimizer77696,
normalizer9024512; reference tensors and teacher unchanged, student finite,
private reference RNGs advanced and resume succeeded with missing original path.
Evidence: logs/camera_reference_20261028/smoke/verification.json.
The separate8-episode student-only playback loaded and completed all first
episodes (0 successes,2 grasps,0 lifts). It verifies compatibility, not accuracy;
no performance improvement is claimed. Task2 complete. Report:
logs/camera_reference_20261028/REPORT.md.

Ruling: preserve this ledger and all raw evidence, with no staging/commits or
publication. Long training is outside this implementation validation; the guide
provides a bounded50-update screening command. No accuracy gain or validated
task-level remedy is claimed from unit tests, collection or smoke checkpoints.

Subsequent user instruction "ok go next" authorized the documented50-update
screen. It completed:181/256 nominal,20/256 wide. All reference tensors stayed
identical and fitting improved, but nominal loss18.75pp failed the3pp gate.
Stop,retain2200,no continuation/1024 confirmation. Optimizer83584,normalizer
9228288,relative50; paired protocol/poses validated. Full evidence under
logs/camera_reference_20261028/screen50/REPORT.md. Protecting references is
implemented; a task-level fix remains unestablished.
