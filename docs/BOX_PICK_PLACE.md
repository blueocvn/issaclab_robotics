# SO-101 upright box transfer to an A4 zone

The new tasks are `IsaacTutorial-Pick-Box-SO101` (state PPO teacher) and
`IsaacTutorial-Pick-Box-SO101-Camera` (wrist-camera distillation student).
The original vial tasks and their checkpoints remain available.

**Current continuation2026-10-27:** the overnight run passed A-E and stopped at
Stage F's convergence limit. Best observed teacher4982 scored98.54% on the
selection audit and98.93% on an unused seed; camera distillation never started.
Next is a user-launched fixed-small-learning-rate PPO refinement from4982.
See [OVERNIGHT_WORKSPACE_RESULTS.md](OVERNIGHT_WORKSPACE_RESULTS.md) for the
verified analysis and exact PowerShell command. The overnight controller is
no longer running; do not restart it from Stage A.

For this PC's WSL distribution, paths and verification evidence, see
[LOCAL_PC_SETUP.md](LOCAL_PC_SETUP.md). The current continuation is
**release-v3**, described below; the lift-v2 command is historical context.

**Latest result:** camera distillation completed; latest student `model_1599.pt`
passed 1,020/1,024 and 1,018/1,024 nominal first episodes on two seed42 audits.
Rare timeouts remain and changed across matching-start repeats. See
[CAMERA_DISTILLATION_RESULTS.md](CAMERA_DISTILLATION_RESULTS.md) for evidence and
the latest playback command. Its refined state teacher `model_2196.pt` passed
3,072/3,072 episodes across seeds42/43/44; that earlier result is recorded in
[RELEASE_V3_REFINEMENT_RESULTS.md](RELEASE_V3_REFINEMENT_RESULTS.md).

**New goal:** pickup within+/-10 cm in X/Y with upright box rotation and a fixed
paper. Frozen-policy screening passed36/1,024 for the state teacher and49/1,024
for the camera student across the full position/yaw distribution. These
checkpoints require wider-distribution training; see
[WORKSPACE_SCREENING_RESULTS.md](WORKSPACE_SCREENING_RESULTS.md) and its proposed
teacher-first training design.

That design is now approved. The optional reset mixture and controlled audit
runner are implemented; commands and validation are documented in
[BOX_WORKSPACE_TRAINING.md](BOX_WORKSPACE_TRAINING.md).
Existing nominal commands keep their original defaults.

**Execution update2026-10-27:** the user subsequently authorized autonomous long
training and shutdown. The staged pipeline is now running under
`logs/overnight_20261027_ppo_camera`; do not launch duplicate manual training.
Its `SUMMARY.md` records the eventual outcome. See the runbook's execution-mode
update for retry limits, monitoring and shutdown behavior.

## Scene and contract

The photo `Image.jpg` defines the orientation: **13 cm long, 4.5 cm thick,
9 cm high**. Pinch across the 4.5 cm thickness. The default box mass is a
**provisional 0.10 kg**, not a measurement of the photographed box.

The box starts upright at A and is transferred to a white **29.7 x 21 cm A4**
paper marker at B. The paper is visual only; the underlying tabletop supports
the box. Newton filtered contact sensing uses a kinematic desk body.

```
               A: pickup                 B: white A4 target
             +-------------+          +-----------------------+
             | 13 x 4.5 cm | -------> |                       |
             +-------------+          |  place upright inside  |
                                      +-----------------------+
                            SO-101 base
```

Coordinates are metres in a world-parallel workcell frame anchored at the
robot base XY (-0.05, 0). X is forward into the table, Y is left. This is not
the yawed USD root-link frame. Defaults are A=(0.22,-0.10), B=(0.22,+0.10).
The box's 13 cm edge and the paper's 29.7 cm edge lie along X. The tabletop
is at environment-local Z=0.035 m.

Success requires a grasp and a 3 cm lift earlier in the same episode, followed
by a released, withdrawn gripper and a box resting upright inside the paper.
Every projected cuboid corner must lie at least 5 mm inside the paper boundary.
Tilt must be at most 10 degrees, linear speed below 0.03 m/s, angular speed
below 0.3 rad/s, and bottom-to-table distance within 3 mm. Table contact is
required. Both jaws must have no box contact, jaw opening must exceed box
thickness by 5 mm, and the pad midpoint must be over 8 cm from the box center.
These conditions must persist for 15 control steps (0.5 seconds).

Episodes start at home and last at most 30 seconds. The initial box position
varies by +/-1 cm and yaw by +/-0.1 rad in training and evaluation. There is
no dependency on the vial reset dataset. Placement anywhere inside the paper
counts; a centered final pose and fixed final yaw are not required.

### Lift-stage reward revision

The first 800-iteration teacher reached bilateral contact but achieved 0/1,024
placements and no 3 cm lifts. Recorded playback showed it pressing the box
against the table at about 10 N (box weight about 1 N), then losing contact.

The revised rewards use a within-episode progression: before clearance, aim
the box and pad midpoint at a point 10 cm above the current box XY; after a
physical 3 cm lift, aim at the paper's resting center. The reward is nonzero
on the table, favors upward motion, and encourages an upright box. The old
height multiplier that zeroed guidance on the table is removed.

Before lift, approach guidance resumes after contact loss. Negative retreat
deltas remain active during contact so sliding away and reapproaching cannot
farm reward; after lift, carry/release rewards take over so final withdrawal
is not penalized. Grasp history requires two consecutive bilateral-contact
steps. Excess table force above the actual box weight plus a 2 N allowance
is penalized, saturating at another 8 N. Live table-force, contact and bottom-
height metrics are logged for monitoring.

This is stage-based reward guidance, not teleporting into easier reset states:
all training and evaluation episodes still start at home. The full placement
criterion is unchanged, and camera/state observations remain 54-dimensional.
Existing box weights can be loaded into a new run, but mean reward before and
after this revision should not be directly compared.

The historical lift-v2 run continued from the first teacher with:

```bash
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101 --num_envs 512 --max_iterations 800 --seed 42 --run_name so101-box-lift-v2 --device cuda:0 --checkpoint logs/rsl_rl/so101_box_state/2026-10-26_18-02-29_so101-box-teacher/model_799.pt --visualizer none presets=newton_mjwarp
```

This creates a separate run and leaves the first teacher checkpoint intact.
Use the new run's final checkpoint for the same exact evaluation.

Record a diagnostic replay from an exported box policy:

```bash
uv run python -m isaaclab_tutorial.tasks.place_box.inspect_policy --policy /absolute/path/exported/policy.pt --num_envs 1 --steps 905 --output logs/box_playback_trace.json --device cuda:0 --visualizer none
```

For viewer frames, use the NVIDIA graphics exports below, change to
`--visualizer newton`, and add `--frames logs/box_playback_frames`. The trace
records pad locations, forces, actions, milestones and episode IDs, retaining
terminal state before automatic reset and all episode boundaries.

### Release-v3 continuation

Lift-v2 reached grasp and physical lift in all 1,024 audited episodes but never
completed placement. Playback showed the box held over the paper with closed
jaws, about 13 degrees of tilt, while repeating held-pose rewards continued to
pay. The continuation checkpoint is
`logs/rsl_rl/so101_box_state/2026-10-26_18-39-43_so101-box-lift-v2/model_1598.pt`.

Release-v3 replaces repeating transport/release bonuses with bounded potentials.
Their progress terms are `0.995 * current_potential - previous_potential`, matching
PPO's discount. First samples after each selected environment's reset give zero;
other environments keep their history. Negative changes are retained when
contact, containment or support is lost, so stationary holds and release/regrasp
cycles cannot repeatedly earn positive progress.

Release potential is enabled after physical lift, with all box corners inside
the paper, bottom within 3 mm of the tabletop and actual table contact. Initial
opening allows up to 25 degrees of tilt so the observed held pose can settle.
Up to 0.25 potential comes from physical jaw opening, another 0.25 from contact
cessation with sufficient jaw gap, and the final 0.5 from withdrawal while
released and upright within 10 degrees. Commanded opening without jaw motion
does not earn opening credit. **The final success criterion is unchanged.**
Transport/release weights are 2/10; success weight remains 200. State actor,
critic and camera teacher inputs remain 54-dimensional, preserving box
checkpoint compatibility.

Training publishes `on_target`, `released`, `withdrawn` and `upright` rates.
These are live batch fractions, not exact episode success rates. `on_target`
requires containment and physical table support; the other stage checks are
conditioned on it. `released` adds no jaw contacts and sufficient gap;
`withdrawn` adds pad distance over 8 cm; `upright` checks the 10-degree limit.
They describe the current/terminal state, not historical-ever milestones.
Metrics are retained on automatic-reset steps and each step publishes a fresh
snapshot so the logger cannot overwrite buffered samples.

The scripted physical release check on the current RTX 5070 passed all eight
first episodes at steps 563–568 with the unchanged success criterion. It runs
the lift-v2 policy for 450 steps, then opens the jaws and withdraws the arm.
This proves physical feasibility with scripted assistance, **not learned release
success**. Evidence is in `logs/local_pc_scripted_release_summary.json` and
`logs/local_pc_scripted_release.json`.

The accepted release-v3 run used this continuation command (retained for reference):

```bash
uv run --locked isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101 --num_envs 512 --max_iterations 400 --seed 42 --run_name so101-box-release-v3 --device cuda:0 --checkpoint logs/rsl_rl/so101_box_state/2026-10-26_18-39-43_so101-box-lift-v2/model_1598.pt --visualizer none presets=newton_mjwarp
```

This creates a separate run and retains lift-v2. The 400 updates are additional
updates; global numbering continues from the saved checkpoint. Choose the new
run's actual numerically greatest checkpoint for evaluation. Monitor lift
preservation and all four stage metrics, then run the exact 1,024-episode audit
below. Camera distillation waits for at least 90% full placement success.

## Verify physics, grasp and camera

On the current PC, run inside WSL distribution `Ubuntu` from
`/mnt/d/work/robotics/IsaacLabTutorial`:

```bash
uv run isaaclab zero_agent --task IsaacTutorial-Pick-Box-SO101 --num_envs 8 --max_steps 100 --device cuda:0 --visualizer none presets=newton_mjwarp

uv run python -m isaaclab_tutorial.tasks.place_box.check --grasp-probe --device cuda:0 --visualizer none

uv run python -m isaaclab_tutorial.tasks.place_box.check --camera --steps 10 --device cuda:0 --visualizer none
```

The diagnostic prints `SO101_BOX_CHECK`. The grasp probe uses actual robot
forward kinematics to find jaw poses, checks A/lift/B reach against joint
limits, then closes and lifts under gravity/contact physics. It never
teleports the box during the grasp/lift. Its initial robot teleport makes it
an isolated grasp feasibility check, not proof of a collision-free approach
or full A-to-B placement policy. Its exit code is nonzero on failure.

Add `--sim2real` to the check commands to verify transfer randomization.

To inspect the scene in a window:

```bash
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
uv run isaaclab zero_agent --task IsaacTutorial-Pick-Box-SO101 --num_envs 1 --device cuda:0 --visualizer newton presets=newton_mjwarp
```

The two exports request NVIDIA D3D12 OpenGL in WSLg. They were verified on the
source PC; the current PC's headless camera renderer has been verified
separately. CUDA physics is selected independently with `--device cuda:0`.

## Train a fresh box teacher

```bash
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101 --num_envs 512 --max_iterations 800 --seed 42 --run_name so101-box-teacher --device cuda:0 --visualizer none presets=newton_mjwarp
```

Checkpoints go to `logs/rsl_rl/so101_box_state/`. Vial weights are not a box
teacher. An 800-iteration budget is an initial experiment, not a guarantee of
convergence. Use exact full-episode evaluation before accepting a teacher.

Layout and physical parameters can be overridden, for example:

```bash
uv run isaaclab zero_agent --task IsaacTutorial-Pick-Box-SO101 --num_envs 8 --max_steps 100 --visualizer none presets=newton_mjwarp 'env.pickup_xy=[0.22,-0.10]' 'env.target_xy=[0.22,0.10]' env.paper_yaw=0.0 env.box_mass=0.1 env.box_friction=0.7
```

The custom parameters are applied before scene construction, so visuals,
collisions, reset poses and evaluation use the same layout.

## Evaluate and watch a checkpoint

Set CKPT to the chosen **box** checkpoint, then run the exact 1,024-episode audit:

```bash
CKPT=/absolute/path/to/so101_box_state/RUN/model_799.pt
SO101_EVALUATION_EPISODES=1024 SO101_EVALUATION_OUTPUT=logs/box_evaluation.json uv run isaaclab play --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101 --num_envs 1024 --device cuda:0 --checkpoint "$CKPT" --visualizer none --external_callback isaaclab_tutorial.tasks.place_box.evaluation.install_episode_counter presets=newton_mjwarp
```

The callback counts each environment's first episode exactly once, prints
`SO101_BOX_EVAL_RESULT`, and writes individual outcomes when an output path
is supplied. It reports grasp, lift, placement, success, dropped-box,
out-of-bounds, robot-instability and timeout rates, plus terminal `on_target`,
`released`, `withdrawn` and `upright` rates and per-episode flags. The acceptance target is
at least 90% full placement success over 1,024 episodes.

For continuous visual playback, use the NVIDIA graphics exports above:

```bash
uv run isaaclab play --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101 --num_envs 8 --device cuda:0 --checkpoint "$CKPT" --visualizer newton presets=newton_mjwarp
```

For a short visible audit, combine `--visualizer newton`, `--num_envs 8`, the
box evaluation callback, and `SO101_EVALUATION_EPISODES=8`.
Do not add `--deterministic`: playback policy actions already are deterministic;
that flag also enables physics determinism with a known buffer issue on this pin.

## Camera distillation and transfer preparation

After the box teacher passes evaluation:

```bash
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101-Camera --num_envs 128 --max_iterations 1600 --seed 42 --run_name so101-box-camera-student --device cuda:0 --checkpoint "$CKPT" --visualizer none presets=newton_mjwarp,newton_renderer
```

The student sees two 64 x 48 wrist RGB frames and joint feedback/history; the
teacher sees the same 54-dimensional state as the state PPO actor. The
frozen teacher labels the student's visited states with clipped actions.
Student checkpoints go to `logs/rsl_rl/so101_box_camera_distillation/`.
Evaluate with the camera task, camera checkpoint and
`presets=newton_mjwarp,newton_renderer`, using the same box audit callback.

For sim-to-real, first continue the accepted state teacher with
`presets=newton_mjwarp,sim2real`. Evaluate that adapted checkpoint, then distill
the camera student with `presets=newton_mjwarp,newton_renderer,sim2real`.
Both tasks share mass/friction, actuator, joint and encoder randomization;
the camera variant also varies appearance, camera mount and image formation.
Always use matching presets for train, evaluation and export.

Measure the real box mass and calibrate real A/B locations before deployment.
Keep the A4 paper flat at the calibrated B location; arbitrary paper detection
is not implemented. Follow the existing calibration, joint-map, export and
read-only dry-run workflow in `SIM2REAL.md`. Export from the **box camera task**,
verify its saved action scales/history, then conduct supervised trials.
