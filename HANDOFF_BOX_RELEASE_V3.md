# Codex handoff — SO-101 box pick-and-place, release-v3

## Stop state and working instructions

The user requested: **Stop and create a handoff; they will continue on their PC.**
Work is stopped. No training, playback or pytest processes were running at the
final process check. Full release-v3 retraining has NOT been started.

The user prefers to launch long training themselves. Short simulation checks
and two-iteration integration checks were performed by the agent. Do not start
a long run automatically or terminate another run without the user's instruction.

## Environment

- Windows workspace: `E:\robotics\IsaacLabTutorial`
- WSL workspace: `/mnt/e/robotics/IsaacLabTutorial`
- WSL distribution: **Ubuntu-24.04** (not simply Ubuntu)
- Branch: **feat/so101-box-placement**, based on commit `7efef77`
- Python 3.12, existing uv environment, Isaac Lab 3.0, Newton MJWarp, RSL-RL PPO.
- uv binary in non-login shells: `/home/apc/.local/bin/uv`
- Detected GPU is **RTX 5060 Ti 16GB**, despite earlier conversation saying 4060 Ti.
- WSL access from the agent required approved/elevated tool execution.
- Do not reinstall the working environment unless necessary.

For NVIDIA graphics in WSLg:

```bash
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
```

Without GALLIUM_DRIVER, this machine selected llvmpipe CPU rendering (~3 FPS).
With both settings, glxinfo reported NVIDIA D3D12 and Accelerated: yes.
CUDA/OpenGL direct interoperability is unavailable in this WSL setup; Newton
falls back to copies. That warning did not prevent successful viewing/capture.

## Task agreed with user

- Task: pick the upright cardboard box at A and place it in the white A4 zone B.
- Reference orientation: `Image.jpg` in repository root.
- Box: **13 cm length x 4.5 cm thickness x 9 cm height**.
- Gripper closes across the 4.5 cm thickness.
- Paper: **29.7 x 21 cm**, flat visual marker; desk provides collision support.
- Anywhere entirely inside the paper counts; exact centering/yaw not required.
- A/B default workcell coordinates: A=(0.22,-0.10), B=(0.22,+0.10) metres.
  Workcell axes are world-parallel and anchored at robot base XY=(-0.05,0).
  Thus single-env scene B is XY=(0.17,0.10). Parallel env traces are in world
  coordinates: subtract the environment origin before comparing locations.
- Tabletop Z=0.035 m. Box mass 0.10 kg is provisional, not a real measurement.
- Original vial tasks must remain available.

Success rules remain unchanged:

1. Sustained bilateral grasp and a physical lift of box bottom >=3 cm.
2. All projected cuboid corners inside the paper with a 5 mm margin.
3. Upright within 10 degrees, bottom within 3 mm of table, actual table contact.
4. No jaw/box contacts; gap >5 cm; pad midpoint >8 cm from box center.
5. Linear speed <0.03 m/s and angular speed <0.3 rad/s for 15 control steps
   (0.5 seconds at 30 Hz; physics 120 Hz).

Keep paper at calibrated B for deployment. Arbitrary paper detection is not
implemented. Measure real box mass and calibrate real A/B before hardware use.

## Code and Git state

**Work is uncommitted. Many feature files are untracked.** A plain git diff
does not include those files. If transferring PCs, copy the whole workspace
or explicitly include all new source/tests/docs and the checkpoint files.

New package: `src/isaaclab_tutorial/tasks/place_box/`

- `__init__.py`: registers state and camera tasks.
- `env.py`: applies CLI layout changes after preset resolution, before scene construction.
- `env_cfg.py`: scene, controls, observations, events, rewards, terminations.
- `geometry.py`: full cuboid/paper geometry and unchanged placement criterion.
- `progress.py`: partial-reset-safe milestones; grasp requires 2 consecutive contact steps.
- `mdp.py`: observations, resets, rewards, success and stage diagnostics.
- `agents.py`: separate box PPO and camera distillation configurations.
- `evaluation.py`: exact first-episode-per-env audit, JSON output.
- `check.py`: CUDA/camera/physical grasp feasibility checks.
- `inspect_policy.py`: TorchScript playback trace, optional viewer frame capture,
  pre-reset terminal snapshots and scripted opening/withdrawal diagnostic.

Tests: `tests/test_box_geometry.py`, `test_box_config.py`,
`test_box_evaluation.py`, `test_box_playback.py`, `test_box_rewards.py`.
`tests/test_config.py` was adjusted to scope its original registration assertion
to vial tasks, allowing the new registrations. README links the box guide.

Guides: `docs/BOX_PICK_PLACE.md`, `docs/BOX_TASK_PROGRESS.md`.
The guide documents lift-v2; the new release-v3 changes still need documenting.

Pre-existing dirty files include demo media and workshop desk/rack/vial USD
assets, plus untracked Image.jpg. These predated the box implementation;
do not revert or discard them. No branch merge/push was performed.

## Training history and known checkpoints

### Original vial teacher (retain)

`logs/rsl_rl/so101_vial_state/2026-10-26_16-03-43_so101-state-teacher/model_799.pt`

### Box teacher v1: grasp but no lift

`logs/rsl_rl/so101_box_state/2026-10-26_18-02-29_so101-box-teacher/model_799.pt`

800 iterations, 512 envs. Exact 1,024-episode audit:
100% grasp history, 0% lift, 0% placement, 100% timeout.
Playback pressed box into table at 10–11 N versus ~0.98 N box weight, then
lost contact. Approach reward was permanently disabled by latched grasp;
transport shaping was zero on table.

### Box lift-v2: grasp/lift/transport learned; release stuck

**Use this checkpoint for continuation:**

`logs/rsl_rl/so101_box_state/2026-10-26_18-39-43_so101-box-lift-v2/model_1598.pt`

800 additional PPO updates resumed from v1, seed42, 512 envs. RSL-RL continued
global iteration numbering from799 to1598.

Exact audit of final checkpoint:

- 1,024 episodes
- 100% grasp history
- 100% physical lift
- 0% completed placement
- 100% timeout
- No dropped/out-of-bounds/unstable robot terminations

Artifacts:

- `logs/box_lift_v2_evaluation.json` and corresponding .log
- `logs/box_lift_v2_playback_trace.json`
- `logs/box_lift_v2_playback_frames/step_0899.png` (final held-on-paper pose)

Single-env inspected final pose: box within9 mm of B center, full footprint
inside paper, both jaws contacting, pad midpoint2.85 cm from box center,
tilt13.4 degrees. Policy kept closing gripper. Training mean reward rose to7.75
despite zero success because held-pose transport/release bonuses kept paying.

Lift-v2 changes already verified before this handoff:
live-contact approach recovery before lift, no reward-farming retreat cycles,
vertical lift goal before B transport, downward-pressure cost, upright shaping,
2-step grasp, unchanged54D actor/teacher observations.

## Scripted release feasibility — PASSED 8/8

Before release-v3 behavior changes, ran the final lift-v2 TorchScript policy
for450 control steps, then held arm/opened jaws for60 steps, then slowly
interpolated the arm toward home over180 steps, keeping gripper open.
No box teleport, gravity change, success relaxation or physical parameter changes.

**All8 first episodes succeeded**, at steps564–568, using the unchanged task
criterion. This is scripted assistance, NOT learned-policy success.

Artifacts: `logs/box_scripted_release_check.json` and .log.

Reproduce:

```bash
cd /mnt/e/robotics/IsaacLabTutorial
uv run python -m isaaclab_tutorial.tasks.place_box.inspect_policy \
  --policy logs/rsl_rl/so101_box_state/2026-10-26_18-39-43_so101-box-lift-v2/exported/policy.pt \
  --num_envs 8 --steps 899 --release-at 450 \
  --output logs/box_scripted_release_check.json \
  --device cuda:0 --visualizer none
```

## Release-v3 changes implemented immediately before stopping

The user approved correcting the release stage. The following code is present:

1. `geometry.placement_geometry` shares containment/upright/table-proximity logic.
   `placement_mask` still enforces exactly the original success rules.
2. `transport_reward` and `release_reward` now provide bounded potentials,
   consumed by `TransportProgressReward` and `ReleaseProgressReward`.
3. Both progress classes compute **0.995 * current_potential - previous_potential**.
   First sample after a partial reset gives0. Negative changes are retained:
   stationary holds pay no positive repeated reward; release/regrasp cycles
   cannot farm reward. Discount0.995 matches the existing PPO gamma.
4. Release potential applies only after lift, with full box inside paper,
   table proximity AND actual table contact. Up to25-degree tilt is allowed
   for initial opening so the observed13-degree held pose can settle under
   gravity. **Final success and full withdrawal credit still require10 degrees.**
5. Potential allocation: up to0.25 for physical jaw opening, another0.25 for
   actual contact cessation with sufficient gap, final0.5 for withdrawal while
   released AND upright. Opening alone cannot earn full release credit.
6. Config now uses transport-progress weight2.0 and release-progress weight10.0.
   Success weight200 and success definition remain unchanged.
7. Added live/terminal `on_target`, `released`, `withdrawn`, `upright` diagnostics.
   Exact evaluation now includes corresponding rates and per-episode flags.
   These are terminal stage checks, not historical-ever flags.
8. Observations remain **54-dimensional**. Existing box teacher checkpoint loads.

## Verification actually completed

- Last complete suite BEFORE release-v3: **119 passed, 2 skipped**.
  Optional missing dependencies: cv2 and leapp. Existing TorchScript warnings.
- Release reward regressions first failed as intended: **7 failures**.
- After progress-potential/geometry implementation:
  **22 reward+geometry tests passed**.
- Audit-metric regression first failed as intended (on_target_rate absent).
  The code fix was subsequently implemented, but that test has NOT been rerun.
- Two-iteration PPO integration using model_1598.pt completed successfully with
  finite losses, new progress rewards and all four new stage metrics present.
  Log: `logs/box_release_v3_integration.log`.
  Run name: `box-release-v3-integration` (smoke only, not trained policy).
- Ruff ran before the last metric edits, fixing one issue. Final lint/full
  regression run and final code review remain pending due user's stop request.
- No claim of learned release success has been made.

## Next steps on the user's PC

1. Preserve/copy all untracked feature files and the lift-v2 checkpoint.
2. Read this handoff and relevant sources; avoid reinstalling a working env.
3. Run final verification:

```bash
cd /mnt/e/robotics/IsaacLabTutorial
uv run ruff check src/isaaclab_tutorial/tasks/place_box tests/test_box*
uv run pytest -q
```

4. Review release-v3 code, especially partial reset, negative potential deltas,
   actual contact/support gating, unchanged success rules, audit terminal
   snapshots and54D checkpoint compatibility. Fix any discovered defects
   before a long run. Re-run scripted physical release if needed.
5. Let the USER launch a separate release-v3 continuation. Suggested initial
   budget400 additional updates, keeping512 envs for comparability:

```bash
uv run isaaclab train \
  --rl_library rsl_rl \
  --task IsaacTutorial-Pick-Box-SO101 \
  --num_envs 512 --max_iterations 400 --seed 42 \
  --run_name so101-box-release-v3 --device cuda:0 \
  --checkpoint logs/rsl_rl/so101_box_state/2026-10-26_18-39-43_so101-box-lift-v2/model_1598.pt \
  --visualizer none presets=newton_mjwarp
```

This command is proposed; **it has not been executed**. Use the run's actual
numerically latest checkpoint, not lexicographic filename sorting or a
hardcoded assumed final index. Global iteration numbering continues.

6. Monitor lift preservation plus on_target/released/withdrawn/upright metrics.
   Evaluate periodically; use full1,024-episode audit for acceptance:

```bash
CKPT=/absolute/path/to/new/run/model_XXXX.pt
SO101_EVALUATION_EPISODES=1024 SO101_EVALUATION_OUTPUT=logs/box_release_v3_evaluation.json \
uv run isaaclab play --rl_library rsl_rl \
  --task IsaacTutorial-Pick-Box-SO101 --num_envs 1024 --device cuda:0 \
  --checkpoint "$CKPT" --visualizer none \
  --external_callback isaaclab_tutorial.tasks.place_box.evaluation.install_episode_counter \
  presets=newton_mjwarp
```

Do NOT add `--deterministic`: policy inference already is deterministic;
that flag also toggles physics determinism with a known collision-buffer issue.

7. Camera distillation waits until state teacher achieves at least90% full
   placement success. Existing box camera runs are smoke checks only.
   Then follow `docs/BOX_PICK_PLACE.md` for camera training and matching
   sim2real adaptation. Do not deploy a smoke checkpoint to hardware.

## Useful progress/diagnostic files

- `logs/box_release_rewards_red.log`, `box_release_rewards_green.log`
- `logs/box_release_metrics_red.log`
- `logs/box_release_v3_integration.log`
- `logs/box_scripted_release_check.json`
- `logs/box_lift_v2_evaluation.json`
- `docs/BOX_TASK_PROGRESS.md` (older ledger; this handoff is newer)

No full retraining, final lint/test sweep, commit, merge, push or hardware
operation should be inferred from this handoff.
