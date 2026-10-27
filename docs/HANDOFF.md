# SO-101 box task handoff

Updated **2026-10-27, Asia/Saigon**. This is the current continuation document.
Completed plans and superseded run reports were consolidated here. Their raw
evidence remains under `logs/`; previous documents remain in Git history.

## Current state: evaluate checkpoint 5381 next

The 400-update Stage F PPO refinement has **finished**. Latest checkpoint:

```text
logs/rsl_rl/so101_box_state/2026-10-27_03-19-50_so101-box-workspace-F-refine/model_5381.pt
```

It resumed teacher 4982 with 512 environments, a fresh optimizer, fixed learning
rate 0.00005, full workspace resets and a 20% nominal mixture. All recorded
scalars are finite; loss samples cover iterations 4982–5381. The last 50
training success estimates averaged 99.14%, including nominal starts and
stochastic actions. **No independent acceptance audit of 5381 has run yet.**

Next: evaluate 5381, compare against 4982, then check the four final acceptance
populations. Do not start another PPO run or camera distillation from training
metrics alone. Commands and gates are in
[BOX_WORKSPACE_TRAINING.md](BOX_WORKSPACE_TRAINING.md).

At the latest process check, no training, playback or overnight controller was
active. The overnight run ended at Stage F's convergence limit. Its helper
requested Windows shutdown at 03:16 on 2026-10-27.

## User instructions

- Supply commands for further long training; the user will launch them.
  Short diagnostics/evaluations are allowed within the task.
- The earlier autonomous training/shutdown permission applied to the completed
  overnight job. Do not revive it or schedule shutdown without a new request.
- Preserve the original vial tasks, checkpoints and physical success contract.
- No hardware operation has been authorized. Do not deploy or move the robot.
- Use **WSL Git** for this private GitHub repository; the user confirmed access.

## Repository and runtime

Repository: [blueocvn/issaclab_robotics](https://github.com/blueocvn/issaclab_robotics),
private, branch `main`. The publication includes code, 179 training checkpoints,
exported policies, run configs and evaluation evidence. Model binaries use Git LFS.

| Purpose | Windows | WSL |
| --- | --- | --- |
| Git publication checkout | `D:\work\robotics\issaclab_robotics` | `/mnt/d/work/robotics/issaclab_robotics` |
| Existing runtime with working `.venv` | `D:\work\robotics\IsaacLabTutorial` | `/mnt/d/work/robotics/IsaacLabTutorial` |

These are separate checkouts. **Push from the publication checkout.** The runtime
checkout retains its original upstream remote/history and uncommitted feature
files; do not reset it or push to that upstream. Synchronize intended changes
between checkouts. The publication checkout has no copied virtual environment.

WSL distro: **`Ubuntu`**, Ubuntu 24.04. GPU: RTX 5070, 12 GB. Python 3.12;
Isaac Lab 3.0.0 at `d061bf35a58273ef0cc8aceae76392166f9b7d15`, Torch 2.11+cu130,
Newton 1.6.1, Warp 1.17 and RSL-RL 5.5.1. See
[LOCAL_PC_SETUP.md](LOCAL_PC_SETUP.md).

Latest full verification: **174 tests passed, 2 optional skips** (`cv2`, `leapp`),
Ruff clean. Model hashes and fresh remote downloads of teacher 5381 and camera
1599 were verified. This is runtime/code verification, not policy acceptance.

## Task and goal

Pick an upright **13 × 4.5 × 9 cm** box and place it upright inside a fixed A4
zone. Pinch across the 4.5 cm thickness. Nominal starts vary by ±1 cm XY and
±0.1 rad yaw. Current goal: **±10 cm XY and full upright yaw**, paper fixed.
Starting on a side or upside down is outside this task.

World-parallel workcell axes are anchored at robot-base XY=(-0.05, 0). Pickup
A=(0.22,-0.10), paper B=(0.22,+0.10), tabletop Z=0.035 metres. Requested box-center
bounds: X=[0.12,0.32], Y=[-0.20,0.00]. Subtract 0.05 m from workcell X for the
single-environment scene coordinate. These are center bounds, not footprint bounds.

Success requires sustained bilateral grasp, a real 3 cm lift, all projected
corners at least 5 mm inside the paper, tilt≤10°, actual table support with
bottom within 3 mm, no jaw contacts, jaw gap>50 mm, pad midpoint>80 mm from
the box, speeds<0.03 m/s and<0.3 rad/s, held for 15 control steps. Control is
30 Hz, physics 120 Hz; timeout 30 seconds. Do not relax these requirements.

## Model roles and observed results

Full paths: [MODEL_INDEX.md](../MODEL_INDEX.md). Hashes:
[checkpoint_manifest.json](../checkpoint_manifest.json).

| Model | Role and evidence |
| --- | --- |
| **5381** | Latest Stage F refinement; finished, independent acceptance pending |
| **4982** | Best observed overnight selection score: 1,009/1,024 (98.54%); additional seed45 screen: 1,013/1,024 (98.93%); below full workspace gates |
| **2196** | Accepted nominal state teacher: 3,072/3,072 across seeds42/43/44; not wider-workspace acceptance |
| **1599** | Nominal camera student: 2,038/2,048 over two matching-start audits; residual failures; not a wider-workspace student |

The overnight controller completed 16 blocks of 200 PPO updates and passed A–E.
Stage F's eight scores on the same 1,024 seed43 starts were
845, 960, 982, 987, 1002, 1009, 1004, 1006. Model 4982 had 15 timeouts:
4 no grasp, 3 grasped without lift, 8 incomplete after lift. Its weakest spatial
bin was 35/42 (83.33%); many failures were near ±180° yaw. The additional seed45
screen's weakest spatial bin was 33/38 (86.84%).

**Wider-workspace camera training has not started.** Generic `population=stage`
status `pass` means ≥95%; Stage F promotion additionally needs ≥99% overall
and ≥95% in every spatial/yaw bin. A stage-screen pass is not final acceptance.

## Code and known constraints

Package: `src/isaaclab_tutorial/tasks/place_box/`.

- `env_cfg.py`, `env.py`: shared configuration/physics, layout overrides and
  preservation of pre-reset metric snapshots.
- `mdp.py`, `geometry.py`, `progress.py`: 54D state, 6D actions, physical
  rewards/success, footprint geometry and partial-reset-safe history.
- `workspace.py`: curriculum, nominal mixture, independent reset RNG,
  controlled 200-case grid and saved-pose replay.
- `evaluation.py`: exact first episode per environment; starts and provenance
  captured before automatic reset.
- `workspace_audit.py`, `workspace_report.py`: separate grid/random/nominal/
  regression populations, hash/provenance checks and coverage gates.
- `workspace_pipeline.py`, root `run_workspace_overnight.ps1`: optional bounded
  automation and Windows power control; currently inactive, not the next workflow.
- `check.py`, `inspect_policy.py`: feasibility/camera/trajectory diagnostics.
  Scripted or assisted release is not learned-policy success evidence.

Do not add `--deterministic`: playback actions already are deterministic; that
flag changes physics and triggers a known collision-buffer issue on this pin.
Use matching task/presets for training, evaluation and export. Do not select
smoke checkpoints as trained policies.

## Remaining camera and transfer work

After wider teacher acceptance, adapt/evaluate sim-to-real physics, then
distill/evaluate a matching camera student. Camera1599 embeds its old teacher;
loading it does not automatically adopt a new teacher. Initialize new distillation
from the accepted PPO teacher unless a student-warm-start method is verified.

Current variation covers solid colors, camera mount and image formation;
**printed patterns/logos/textures are not randomized**. The user's real box has
patterns and small mesh differences. Add/validate appearance coverage during
camera preparation, confirming what actually renders in wrist images.
Matching dimensions alone does not establish transfer. Mass 0.10 kg is
provisional; measure the box and calibrate camera/A/B before hardware use.
The A4 stays fixed; arbitrary paper detection is not implemented.

[SIM2REAL.md](SIM2REAL.md) is the original vial reference. Its vial-specific
action scales, layout and commands must not be copied blindly to box deployment.

## Evidence

- `logs/overnight_20261027_ppo_camera/`: outcome, journal and per-block audits.
- `logs/overnight_workspace_analysis.json`: recomputed counts, matched starts,
  failure classification and finite-scalar checks.
- `logs/workspace_4982_holdout45_20261027/`: additional 4982 screen.
- `logs/box_workspace_F_refine_training_summary.json`: completed refinement.
- `logs/workspace_overnight_final_tests.log`: 174-pass runtime suite.

New generated outputs remain ignored by default; explicitly stage intended new
artifacts when publishing them. Preserve raw evidence and model binaries.
