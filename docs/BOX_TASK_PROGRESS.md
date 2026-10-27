# Box task implementation ledger

Approved design: 13 x 4.5 x 9 cm upright box, grasp across 4.5 cm,
place released and stable entirely inside an A4 paper with 5 mm margin.
Retain the vial task. Build state PPO, camera distillation, evaluation and diagnostics.

- Work runs on feat/so101-box-placement in the existing checkout to retain
  its working uv environment and user assets. No dependency reinstall.
- Default mass 0.10 kg is provisional. Real mass and A/B calibration required.
- Implement and verify the task before expensive training. Launch training
  only after a physical grasp/reach feasibility probe passes; report any blocker.
- Base-relative workcell coordinates use world-parallel X forward / Y left,
  anchored at robot position. The authored robot root retains its asset yaw.
- Success requires a real grasp and lift in the same episode, not merely
  a box spawned on B. Placement hold is 15 control steps at 30 Hz.
- Baseline: 95 tests passed, 2 optional dependency skips (cv2 and leapp).
- Geometry/config/accounting: 12 new tests passed. Ruff passes.
- CUDA smoke: 8 environments, 100 steps, 54D policy/critic, 6D actions.
- Camera smoke: valid stacked RGB [8,6,48,64] on CUDA.
- Physical probe: bilateral grasp maintained, box bottom lifted 0.04916 m;
  A/lift/B pad IK errors 1.0/3.6/4.2 mm. Probe uses an initial robot teleport;
  it does not claim a full collision-free approach or learned placement.
- Ruling: use a kinematic desk body in the box task because Newton cannot
  filter contact against the original body's static-only desk geometry.
- Ruling: grasp probe requests 10 cm pad lift to account for measured arm
  deflection/object tilt; task lift requirement remains 3 cm.
- PPO integration: 2 iterations completed and model_1.pt loaded for an exact
  8-episode audit. As expected for a smoke checkpoint: 0/8 successes, 8 timeouts.
- Camera distillation integration: 1 iteration loaded that teacher and saved
  a camera checkpoint. This verifies plumbing, not a trained camera policy.
- Final code review identified nominal friction overridden during Newton
  MODEL_INIT; fixed with a nominal startup material event. Regression test
  observed failing first and then passed in the full suite.
- Final suite: 110 passed, 2 optional-dependency skips; Ruff clean.
- Camera sim2real smoke passed with finite [8,6,48,64] observations.
- User requested to run full training themselves. Cancelled the newly submitted
  pipeline before training output/iterations; no automatic subsequent stages.
  Run the documented 512-environment, 800-iteration command manually.

## Lift-stage diagnosis and revision

- User's full teacher finished 800 iterations. Exact audit: 0/1024 lifts or
  placements; 1024 bilateral-contact milestones and 1024 timeouts.
- Inspected NVIDIA-viewer frames and deterministic trace. At steps91/301 the
  box bottom remained on the table, with 10–11 N table force vs ~0.98 N weight.
  Later the jaws lost contact while the latched grasp flag disabled approach.
- Added tests observed failing for recovery, zero-at-floor reward, lowering
  guidance, pressure cost and one-frame grasps. Implemented staged vertical
  lift -> B lowering guidance, live-contact approach recovery, pressure cost,
  upright shaping and two-step grasp persistence.
- Review found a repeatable approach reward cycle. Regression failed first;
  fixed by retaining negative contact deltas and using unclipped potential
  differences. Disable approach after lift to permit release withdrawal.
- Review also found reset-state contamination in playback telemetry. Added
  a regression test and pre-reset snapshots plus episode-boundary recording.
- Ruling: use stage-based reward guidance first, preserving home-only resets
  and 54D observations; no reset dataset or task-success relaxation.
- Two validation PPO iterations loaded model_799.pt with the revised rewards
  and completed with finite losses. This is plumbing validation, not proof of
  improved learning. Full retraining remains manual at the user's request.
- Final revised suite: 119 passed, 2 optional-dependency skips; Ruff clean.
  Runtime replay across step900 preserved episode0 terminal grasp/geometry,
  then step905 correctly reported episode1 reset state.

## Release-v3 handoff review on the current PC

- Current workspace: `D:\work\robotics\IsaacLabTutorial`, WSL distribution
  `Ubuntu`, RTX 5070 12 GB. Local setup and GPU stepping passed first.
- Reviewed release-v3 signed potentials, partial resets, physical support and
  contact gates, unchanged success, terminal snapshots and 54D compatibility.
- Independent review found terminal stage metrics lost when Isaac Lab replaced
  its log during reset; later calls also overwrote dictionaries buffered by
  RSL-RL. Reproduced both failures before correcting the metric lifecycle.
- `BoxEnv` now retains `Metrics/*` across reset and `BoxSuccess` publishes fresh
  step snapshots. Rewards, physics and task success are unchanged.
- Final box tests: 37 passed. Full suite: 132 passed, 2 optional skips. Ruff clean.
- Current-PC scripted release: 8/8 first episodes succeeded at steps563–568.
  This uses scripted opening/withdrawal and does not establish learned success.
- Two-iteration continuation smoke loaded lift-v2 `model_1598.pt` with512 envs,
  finite losses and all four new stage metrics. Its checkpoint is smoke-only.
- Long400-update continuation remains user-launched from the preserved lift-v2
  checkpoint. Camera distillation waits for>=90% full success on1,024 episodes.
- Detailed evidence and next command: `RELEASE_V3_REVIEW.md` and the release-v3
  section in `BOX_PICK_PLACE.md`.

## Release-v3 trained acceptance result

- User completed the400-update release-v3 run, global iterations1598–1997,
  with512 environments. Latest checkpoint: `2026-10-26_14-21-24_so101-box-release-v3/model_1997.pt`.
- Independent exact1,024-first-episode audit:1,023 full successes (99.9023%),
  100% grasp/lift/on-target/upright, one timeout, no drops/out-of-bounds/unstable
  robot failures. No scripted assistance or physics determinism flag.
- The sole failed episode527 reached the target upright but did not complete
  release/withdrawal. The nominal teacher exceeds the90% acceptance threshold.
- Next stage: user-launched wrist-camera distillation from `model_1997.pt`,
  128 environments,1600 updates, matching nominal renderer/physics presets.
- Results, curves, caveats and launch command: `RELEASE_V3_RESULTS.md`.

## Follow-up on the single failed episode

- User questioned proceeding while one episode still failed. Replayed the same
  1,024-environment configuration and verified episode527's saved initial pose.
- Reproduced its900-step timeout: both jaw contacts approximately1 N, jaw gap
  58.07 mm, pad distance68.86 mm (requires>80 mm), tilt6.39 degrees. It briefly
  released earlier but recontacted before completing withdrawal.
- Recommend pausing camera distillation for an initial200-update PPO refinement
  from `model_1997.pt`, followed by the original audit and additional seeds.
  No long refinement has been started; this is a proposed experiment.
- Physical measurements, trace and manual command: `RELEASE_V3_FAILURE_527.md`.

## PPO refinement accepted; camera stage next

- User completed200 additional PPO updates in
  `2026-10-26_14-59-07_so101-box-release-v3-refine`, iterations1997–2196.
  Latest checkpoint is `model_2196.pt`; all training scalar series are finite.
- Exact first-episode audits: seeds42/43/44 each passed1,024/1,024, totaling
  3,072/3,072 full placements with no timeout or physical failure terminations.
- Verified all original seed42 initial poses match the prior audit. Case527
  now succeeds, releases and withdraws; all earlier successes are retained.
  Seeds43/44 have different starts and also pass.
- Next: user-launched camera distillation from refined `model_2196.pt`, with
  128 environments and1600 updates using the matching nominal renderer preset.
  No further long run was started by the agent.
- Current report, validation data and launch command:
  `RELEASE_V3_REFINEMENT_RESULTS.md`.

## Camera distillation completed

- User completed1,600 student updates with128 environments; latest checkpoint
  is `2026-10-26_15-21-58_so101-box-camera-student/model_1599.pt`.
- Training scalars are finite; first/last50 mean behavior loss decreased87.3%.
- Exact latest-student audits passed1,020/1,024 and1,018/1,024 on matching seed42
  starts. All episodes grasped/lifted; residual failures were timeouts.
- Earlier student1500 passed1,022/1,024 once. Failure IDs varied between latest
  checkpoint repeats, and instrumented replay did not reproduce the first four
  failures. The source of rollout variation remains unisolated.
- Requested next action: visual playback of latest1599. No additional long run
  or hardware operation was started. Report: `CAMERA_DISTILLATION_RESULTS.md`.

## New goal: wider pickup positions and upright rotation

- User selected+/-10 cm in X and Y with box rotation. Working interpretation:
  full upright tabletop yaw, fixed A4 paper.
- Existing evaluator screening, without source behavior changes:
  teacher2196 at+/-5 cm and small yaw110/256; teacher at+/-10 cm/full yaw36/1,024;
  camera1599 at+/-10 cm/full yaw49/1,024. All failures were timeouts.
- Majority of wide failures never grasped. Teacher and camera starts share a
  distribution but do not match per environment despite the common seed42.
- Individual counts, reset bounds and binned totals verified; heatmap generated.
- Proposed next: staged PPO teacher expansion, then matching camera distillation.
  No long training or task behavior changes started. Report:
  `WORKSPACE_SCREENING_RESULTS.md`; design:
  `superpowers/specs/2026-10-26-box-workspace-design.md`.

## Wider-workspace tooling and authorized overnight execution

- Approved design implemented:optional20% nominal reset mixture, A-F range
  stages, deterministic controlled grids, independent audit RNG, saved-pose
  regression and separate population gates.
- Final full suite174 passed,2 optional skips; Ruff passed. Two-update GPU PPO
  check loaded2196, verified finite losses/fresh optimizer and105/407 nominal/
  stage resets. Matched200-case teacher/camera grid starts verified.
- User explicitly authorized long PPO through camera distillation, overriding
  manual-launch preference, and shutdown after completion OR unresolved
  failure. Recoverable execution/convergence issues get bounded retries.
- Durable controller launched in `logs/overnight_20261027_ppo_camera`, starting
  at Stage A from the accepted nominal teacher2196. Windows wrapper keeps the
  PC awake, preserves logs/status and requests graceful shutdown afterward.
- Next inspection:that run's `SUMMARY.md`, `status.json`, `journal.jsonl` and
  sibling `_launcher` logs. A failed outcome does not count as model acceptance.

## Morning analysis and manual continuation —2026-10-27

- Read the completed overnight report:16 PPO blocks,3,200 updates; stages A-E
  accepted. F exhausted its8-block limit. Camera distillation did not start.
  The Windows helper requested shutdown at03:16 local time as authorized.
- All recorded training scalars finite; audit counts recomputed. Best observed
  F6 teacher4982:1,009/1,024, with15 timeouts (4 no grasp,3 no lift,8 incomplete
  after lift). Latest5380:1,006/1,024. All F selection starts match.
- Additional frozen4982 screen on unused seed45:1,013/1,024 (98.93%), weakest
  spatial bin33/38 (86.84%). Strict F gate still unmet; no final acceptance
  or camera claim. This short evaluation has completed; no training started.
- User now wants to launch any further training manually. Prepared400-update
  PPO refinement from4982 with fresh optimizer and fixed0.00005 learning rate,
  preserving other task/training settings. Report/command:
  `OVERNIGHT_WORKSPACE_RESULTS.md`. This is a controlled experiment; if it
  plateaus, inspect failure trajectories before repeating identical blocks.
