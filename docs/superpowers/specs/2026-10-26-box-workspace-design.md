# Wider box pickup area and upright rotation

Status: design approved by the user on2026-10-26; implementation in progress.
Frozen-policy screening is complete. Long wider-distribution training has not
been started by the agent.

## Goal and scope

The user wants the box to start within **+/-10 cm in both X and Y** of the
current pickup point and to be rotated. The working interpretation of rotation
is any upright tabletop yaw in [-pi, pi]; starting on a side or upside down is
outside this proposal. Keep the A4 paper fixed at the existing target.

Pickup center in robot-base workcell coordinates remains (0.22,-0.10) m.
Requested box-center bounds are X=[0.12,0.32], Y=[-0.20,0.00] m. In the scene's
translated frame, subtract0.05 m from X. These are box-center bounds, not the
box footprint. Existing desk dimensions contain the entire upright footprint
throughout this region, including at arbitrary yaw. Robot reachability and
collision-free approach are not yet established for every pose.

Preserve all existing success conditions, box dimensions, action/observation
interfaces, nominal physics, fixed paper, original vial tasks and checkpoints.
The task remains grasp, real lift, upright placement, release and withdrawal;
a start overlapping the paper still needs grasp and lift to succeed.
Hardware transfer and moving paper detection remain separate work.

## Evidence and approach

The agreed direction is to evaluate the existing policy over a wider area,
then expand the state teacher before distilling a matching camera student.

Current nominal teacher2196 passed3,072/3,072 within the old +/-1 cm,
+/-0.1 rad reset distribution. The nominal camera1599 passed2,038/2,048 over
two matching-start audits. These results do not cover the new goal.

Frozen teacher2196 screening:

| Reset distribution | Full successes | Grasp | Lift |
| --- | ---: | ---: | ---: |
| +/-5 cm, yaw +/-0.1 rad | 110/256 (43.0%) | 165/256 | 157/256 |
| +/-10 cm, full upright yaw | 36/1,024 (3.52%) | 260/1,024 | 75/1,024 |

Both tests had only successes/timeouts; no drop, out-of-bounds or unstable-robot
termination. Camera1599 passed49/1,024 (4.79%) across the full region/yaw.
Its starts differ from the teacher's despite the common seed, so these are
distribution-level results, not a paired ranking. Full evidence is recorded in
`docs/WORKSPACE_SCREENING_RESULTS.md`. These are uniform random
starts grouped into spatial/yaw bins after evaluation, not fixed grid points.
Use saved starts and individual episode outcomes as the authoritative evidence.

Approaches considered:

1. **Continue teacher PPO through progressively wider resets, then distill.**
   Recommended: retains the learned release sequence while addressing the
   demonstrated grasp/lift deficit. The user already selected the evaluate,
   improve teacher, then camera direction in conversation.
2. **Widen camera distillation alone.** The frozen teacher is unsuccessful on
   most requested starts, so its labels cannot establish the required behavior.
3. **Restart PPO from scratch across the entire range.** Discards the accepted
   nominal behavior and creates a harder exploration problem; reserve this for
   evidence that continuation cannot learn the new poses.

## Proposed training progression

Resume teacher2196 in a new run. Expand position before adding large yaw:

| Stage | Position half-width | Yaw half-width |
| --- | ---: | ---: |
| A | 2 cm | 0.1 rad |
| B | 5 cm | 0.1 rad |
| C | 10 cm | 0.1 rad |
| D | 10 cm | pi/4 |
| E | 10 cm | pi/2 |
| F | 10 cm | pi |

These stages are an engineering proposal, not an experimentally established
schedule. Advance using episode completion results, not elapsed updates or
instantaneous metric occupancy. Use an initial200-update user-launched block
at each stage, audit, then decide whether to advance or continue. Do not
automatically chain long blocks. Stage checkpoint paths must come from actual
saved files, with explicit resume metadata and optimizer treatment.

Retain nominal experience with a proposed20% nominal /80% current-stage reset
mixture during training. The full-stage audit must use100% the requested test
distribution; mixing easy starts into that score would hide weak coverage.
Make this opt-in so existing nominal training/playback defaults stay compatible.
The evaluation callback must record which distribution generated each start.

Initially retain rewards, observation54D and action6D. If yaw stages fail,
inspect contact/geometry trajectories and feasible gripper orientation before
changing rewards. Do not declare hard poses unreachable based only on policy
failure. Any reachability exclusions must be explicit and reported against the
original requested rectangle, rather than silently resampled away.

## Validation and acceptance

Provide a repeatable workspace audit using a5x5 box-center grid including
rectangle boundaries and8 upright yaw angles spaced45 degrees apart. A run
contains200 position/yaw cases; save initial poses, checkpoint, physics and
renderer settings, seeds, terminations, grasp/lift/placement flags and bounds.
Provide random-jitter evaluations between grid points as a separate test.
Keep the current exact first-episode accounting and pre-reset terminal data.

Proposed advancement gate: at least95% full success on the current stage,
with failure trajectories checked; a stage may still need more training when
failures cluster. Final acceptance proposal: at least99% overall full success,
at least95% in every marginal spatial bin and every yaw bin, and no unreported
or excluded starts. Apply these gates to each population below separately;
never pool boundary-grid scores with easier random starts.

- Fixed grid:200 position/yaw cases, five repetitions per case, seeds45/46/47,
  giving3,000 episodes. This supplies120 trials per spatial point and375 per
  yaw bin across seeds. Repetitions measure rollout variation at the fixed
  cases; they are not3,000 distinct initial poses. All200 joint position/yaw
  case scores must be reported even though gates use marginal bins.
- Uniform random interior: a separate1,024 episodes per seed45/46/47 initially.
  Require at least100 observations in each of the5x5 spatial bins and8 yaw
  bins across seeds. If counts fall short, append predetermined full batches
  of256 starts, independent of outcomes, until coverage is met. Apply the
  same99% overall and95% marginal-bin gates to this population separately.

Also audit the old nominal distribution on seeds42/43/44,1,024 episodes each.
Block teacher promotion below99.9% nominal full success; block student
promotion below99.5%. The retained nominal-case threshold is intentionally
stricter than the current-stage advancement gate. Replay the saved initial
pose of the historical release-failure case527 three times in each policy;
any failure blocks promotion pending analysis rather than being erased by
rerunning until it passes. These thresholds are proposed engineering choices,
not user-established guarantees. Report counts and uncertainty; finite testing
does not establish universal reliability.

Only after the expanded teacher passes, run matching camera distillation.
Use the current student as a candidate initialization if the installed runner
can load that student while explicitly replacing its frozen teacher; verify
loaded weights and teacher identity in a short integration check. Otherwise
use the established teacher-initialized distillation path. Audit the actual
student independently with the same spatial and yaw coverage. Inspect wrist
camera visibility at failed poses before assuming imitation loss is the issue.

The agent may prepare tooling, tests, plots and short integration checks after
design approval. The user launches long PPO/distillation runs, following the
handoff preference. Do not terminate user processes or perform hardware actions.

## Implementation boundaries and remaining decisions

Implement the reset mixture and spatial/yaw audit as opt-in box-task features;
keep nominal defaults intact. Test partial resets, normalized yaw quaternions,
configuration validation, boundary/grid coverage and correct first-episode
counts. Run a short GPU check before supplying long-run commands.

Builder choices: precise config names, batching to fit12 GB VRAM, plotting
style, and internal data structures. Approval is requested for the proposed
progression and finite-test gates above before implementing training changes.
