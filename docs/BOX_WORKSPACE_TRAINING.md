# Wider-position box training

**Latest outcome2026-10-27:** the overnight controller stopped at the Stage F
convergence limit and exited. Stages A-E passed; camera did not start. The user
now launches further training manually. Resume teacher4982 using the refinement
command in [OVERNIGHT_WORKSPACE_RESULTS.md](OVERNIGHT_WORKSPACE_RESULTS.md),
rather than restarting this runbook's initial Stage A command.

**Execution mode updated2026-10-27:** the user authorized autonomous PPO through
camera distillation and Windows shutdown after success or unresolved failure.
The overnight run is `logs/overnight_20261027_ppo_camera`; inspect its
`SUMMARY.md`, `status.json` and `journal.jsonl`. The sibling `_launcher` folder
contains Windows launcher/stdout/stderr logs. The manual commands below remain
reference; do not start a second training run while this controller is active.

The controller trains200-update PPO blocks through A-F, evaluates before
advancing, and requires all four final teacher populations before camera
distillation. Camera starts with1600 updates, then up to two400-update
refinements if needed. PPO execution retries can reduce environments for OOM;
unreadable/nonfinite checkpoints are skipped in favor of valid saved progress.
Convergence attempts are bounded at8 blocks per PPO stage and10 hours total.
On completion or unresolved failure, it saves the outcome/checkpoints and the
Windows helper requests graceful shutdown after a3-minute recovery grace.
The helper keeps the computer awake during training; no permanent power
settings are changed. Unsaved applications can block graceful shutdown.

To explicitly cancel shutdown, create `NO_SHUTDOWN` in the run folder or its
sibling `_launcher` folder. To stop the pipeline as well, create `STOP` in the
run folder. A STOP alone still leads to shutdown, following the user's latest
instruction. All controller subprocesses have their own process groups, so
cleanup targets this run's processes rather than other user sessions.

The approved goal is a box center within **+/-10 cm in X and Y** of the old
pickup point and any upright tabletop yaw. The paper remains fixed. Starting
on a side is outside this task. Existing success/reward/physics settings remain
in force. Baseline evidence: [WORKSPACE_SCREENING_RESULTS.md](WORKSPACE_SCREENING_RESULTS.md).

## Start Stage A: user-launched PPO

Run in PowerShell. This continues the accepted state teacher2196 for200 updates
with512 environments, using80% starts within+/-2 cm and+/-0.1 rad, and20% old
nominal starts. It restores actor, critic, observation normalization and
iteration, with a **fresh optimizer at configured learning rate0.0003**. The
existing adaptive learning-rate schedule remains active.

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/work/robotics/IsaacLabTutorial; uv run --locked isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101 --num_envs 512 --max_iterations 200 --seed 42 --run_name so101-box-workspace-A --device cuda:0 --checkpoint logs/rsl_rl/so101_box_state/2026-10-26_14-59-07_so101-box-release-v3-refine/model_2196.pt --no-resume_optimizer_state --visualizer none presets=newton_mjwarp agent.algorithm.learning_rate=0.0003 env.events.reset_home.params.position_noise=0.02 env.events.reset_home.params.yaw_noise=0.1 env.events.reset_home.params.nominal_fraction=0.2'
```

The authorized overnight controller is now running the equivalent first PPO
block under its own unique run name. Do not launch a duplicate manual run.
Integration smoke checkpoints are labeled `so101-box-workspace-A-smoke` and are not accepted
training results; do not substitute them for the starting teacher above.

After training, select the numerically latest `model_N.pt` inside the actual
`..._so101-box-workspace-A` run, not a checkpoint from a different run and not
simply the most recently modified file. Record that exact path for all audits.

## Stage audit before advancing

In an Ubuntu WSL terminal:

```bash
cd /mnt/d/work/robotics/IsaacLabTutorial
CKPT='logs/rsl_rl/so101_box_state/REPLACE_WITH_ACTUAL_STAGE_A_RUN/model_N.pt'
.venv/bin/python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint "$CKPT" --policy teacher --population stage --stage A --seeds 43 --output-dir logs/workspace_A_stage_audit
```

Set `CKPT` to the actual checkpoint first. Output directories must be new:
the runner refuses to overwrite an existing directory. It launches bounded
play processes, captures their full logs/raw outcomes and writes `manifest.json`
and `summary.json`. The manifest identifies the checkpoint by path and SHA256.
`--dry-run` prints the exact commands without simulation or creating output.
Seed groups are base labels. Each batch uses a distinct, recorded derived seed
for physics and an independent reset RNG, so model initialization does not
change random audit starts. Colliding derived seeds are advanced deterministically
to the next unused seed and recorded in the manifest.

The stage audit uses **100% current-stage starts**, without the20% easy nominal
mixture. Advance only at>=95% full episode success, after inspecting failures
and spatial clusters. Otherwise continue from that stage's actual checkpoint
for another user-launched block and audit again. Training loss and live
occupancy metrics alone are not advancement criteria.

| Stage | position_noise | yaw_noise | Meaning |
| --- | ---: | ---: | --- |
| A | 0.02 | 0.1 | +/-2 cm, +/-5.7 degrees |
| B | 0.05 | 0.1 | +/-5 cm, +/-5.7 degrees |
| C | 0.1 | 0.1 | +/-10 cm, +/-5.7 degrees |
| D | 0.1 | 0.7853981633974483 | +/-10 cm, +/-45 degrees |
| E | 0.1 | 1.5707963267948966 | +/-10 cm, +/-90 degrees |
| F | 0.1 | 3.141592653589793 | +/-10 cm, full upright rotation |

For each next stage, change the run name and these two event overrides in the
training command, and resume the **accepted preceding checkpoint**. Keep
`nominal_fraction=0.2` for training. Use the corresponding `--stage` in audits.
Do not automatically chain stages or train the full rectangle from scratch.

## Final teacher acceptance

After stage F, run these **separate populations** using its exact checkpoint:

```bash
.venv/bin/python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint "$CKPT" --policy teacher --population grid --stage F --output-dir logs/workspace_F_teacher_grid
.venv/bin/python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint "$CKPT" --policy teacher --population random --stage F --output-dir logs/workspace_F_teacher_random
.venv/bin/python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint "$CKPT" --policy teacher --population nominal --output-dir logs/workspace_F_teacher_nominal
.venv/bin/python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint "$CKPT" --policy teacher --population regression --output-dir logs/workspace_F_teacher_case527
```

- Grid:5x5 centers including the rectangle boundaries,8 angles45 degrees apart
  over[-180,180),5 repetitions per case and seed groups45/46/47:3,000 episodes.
  The grid's physical starts match across teacher and camera regardless of
  model initialization. Repeats measure rollout variation, not new poses.
- Random:1,024 uniform starts per seed group45/46/47 initially. Additional
  predetermined256-start streams are appended only if spatial/yaw sample
  counts are insufficient; supplementation does not depend on success rate.
- Nominal:1,024 starts in the old+/-1 cm/+/-0.1 rad range per group42/43/44.
- Regression:three copies of the saved historical case527 start. The source
  defaults to `logs/box_release_v3_1997_evaluation.json`; its pose is saved
  alongside the audit. This is a replay check, not three distinct starts.

Grid and random each require>=99% full success overall,>=95% per marginal
spatial bin and per yaw bin, sufficient counts (at least100 per marginal bin),
and three seed groups. Grid cases/counts/physical starts must also match the
declared layout; all200 joint position/yaw case results are reported. Nominal
teacher acceptance requires>=99.9%; all three case527 replays must succeed.
Do not pool these populations. Reports include Wilson intervals; point-score
thresholds are finite-test criteria, not universal reliability guarantees.

Exit code0 means the specified population passes;2 means failure or insufficient
coverage. A deliberate smoke run with `--max-jobs 1` will ordinarily return2.
A child-process/configuration exception indicates an execution problem and
leaves its log/manifest for diagnosis. Partial/smoke runs cannot establish
final acceptance. Wider initial positions can overlap the fixed target; the
original grasp/lift requirement still applies.

## Camera stage

Only after teacher acceptance, use its exact checkpoint in the established
camera-distillation command, adding the same stage F range and training mixture:

```text
env.events.reset_home.params.position_noise=0.1
env.events.reset_home.params.yaw_noise=3.141592653589793
env.events.reset_home.params.nominal_fraction=0.2
```

The current camera1599 contains its old frozen teacher. Loading it directly
would retain that teacher; do not assume it adopts the new teacher. Until a
short weight-identity check proves a student-warm-start procedure, use the
existing PPO-teacher checkpoint initialization path for the new distillation.
User launches the long camera run. Audit the actual resulting student with
`--policy camera` on the same four populations. Nominal camera threshold is
>=99.5%; broad grid/random and case527 gates match the teacher's.

The wrist camera may not see every requested start from home. Inspect images
and trajectories at failed starts; neither policy failure nor missing visual
coverage by itself establishes mechanical unreachability. Report any
reachability limitation explicitly against the requested region.

## Implementation verification

- Full suite:164 passed,2 optional skips (`cv2`, `leapp`). Ruff passed on the
  changed source/tests; new workspace files pass formatting checks.
- Stage A GPU smoke:512 environments, two PPO updates, loaded teacher2196;
  saved `2026-10-26_16-55-57_so101-box-workspace-A-smoke-fixed/model_2197.pt`.
  All recorded scalars were finite. Optimizer step80 after two updates versus
  88,000 in the source confirms the intended fresh optimizer.
- First GPU reset batch:105 nominal and407 current-stage starts. Nominal
  offsets stayed within1 cm; current-stage offsets stayed within2 cm;
  yaw/quaternion checks passed. Runtime partial resets exposed framework
  int32 IDs; metadata now uses explicit int64 and both paths have regressions.
- Independent review found truncated-audit, grid-repeat identity and derived-seed
  collision gaps. All reproduced before fixes; the full suite above includes
  their regressions and planned-case/configuration binding checks.
- All five audit populations dry-run successfully with the intended initial
  episode budgets and unique actual sampling seeds. Evidence:
  `logs/workspace_commands_verification.json`, `logs/workspace_A_smoke_verification.json`,
  `logs/workspace_final_tests.log`.
- Matched grid smoke:all200 initial poses match within1e-7; frozen teacher4/200,
  camera6/200, both correctly report insufficient coverage on the truncated
  final audit. These checks establish plumbing, not broad policy acceptance.
- Overnight controller:10 tests pass, including process-tree cleanup, bounded
  retries, gate ordering and recovery from interrupted/corrupt saves. Independent
  review's three recovery findings were reproduced and fixed. Full final
  suite174 passed,2 optional skips; Ruff passed. Windows script syntax and the
  keep-awake API were checked without requesting shutdown during verification.
- Actual run launch metadata: `logs/overnight_launch.json`. The Windows helper
  runs independently of this chat and powers off only after the controller
  exits and final logs are retained, per explicit user authorization.
