# Camera data-first experiment

The completed800-update run was screened first. Checkpoint2200 was strongest:
229/256 nominal (89.45%) and80/256 wide (31.25%), compared with1798 at225/256
(87.89%) and57/256 (22.27%). Its wide gain was8.98 percentage points, below
the agreed10-point gate. No candidate qualified for1024-start confirmation.
Use2200 as the common experimental baseline; this is selection evidence,
not full acceptance. [Screen report](../logs/camera_800_gate_20261027_170933/REPORT.md).

The new trainer is opt-in with `--agent rsl_rl_data_first_cfg_entry_point`.
The default box and vial trainers are unchanged. It uses the same wrist crop,
CNN, teacher5381 and physical success criteria. No overhead camera is needed
for this experiment.

Teacher assistance is assigned per episode, staying fixed until termination.
Probabilities by **new experiment updates**:0–49=1,50–149=.5,150–249=.25,
250–349=.1,350 onward=0. The first50 updates are included in the400 budget.
Persistent replay holds at most4096 examples per pre-action phase: approach,
grasp before lift, lift before placement, placed. Minibatches have64 fresh and64
phase-balanced replay samples; empty replay uses128 fresh. Replay is inserted
after optimization, so a rollout cannot replay itself during that update.
Both methods perform128 optimizer steps per update. Normalization sees only
fresh observations. Replay uses CPU FP16 RGB and FP32 proprio/labels, preserving
image noise outside[0,1]. A full replay adds about600MB per checkpoint.

Checkpoints preserve replay, cursors, counters and private RNGs. Legacy camera
checkpoints start relative update0 with empty replay. Resuming resets simulator
episodes and draws new controllers; this does not promise identical physics
trajectories to an uninterrupted run. Playback always uses the student alone.

## Run the controlled training yourself

Use the Ubuntu terminal in WSL. Paste this setup once; it creates no script:

```bash
cd /mnt/e/issaclab_robotics
BASE='logs/rsl_rl/so101_box_camera_distillation/2026-10-27_16-26-29_so101-box-workspace-camera-1798-lr1e-4-seed43/model_2200.pt'
TAG=$(date +%Y%m%d_%H%M%S)
COMMON=(--rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101-Camera --num_envs 128 --seed 43 --device cuda:0 --max_iterations 200 --resume_optimizer_state --visualizer none)
OVERRIDES=(presets=newton_mjwarp,newton_renderer agent.algorithm.learning_rate=0.0001 agent.init_at_random_ep_len=false env.events.reset_home.params.position_noise=0.1 env.events.reset_home.params.yaw_noise=3.141592653589793 env.events.reset_home.params.nominal_fraction=0.2 env.events.reset_home.params.sampling_seed=43)
CTRL1="logs/rsl_rl/so101_box_camera_distillation/${TAG}_control_b1"
CTRL2="logs/rsl_rl/so101_box_camera_distillation/${TAG}_control_b2"
DATA1="logs/rsl_rl/so101_box_camera_data_first/${TAG}_data_b1"
DATA2="logs/rsl_rl/so101_box_camera_data_first/${TAG}_data_b2"
printf 'Experiment tag: %s\n' "$TAG"
```

Run each command separately. Both arms start with the same2200 weights and
optimizer. Keep this terminal open so the variables remain set. Record `TAG`
if restarting the terminal. Every run uses a fresh output folder; do not reuse
the tag after a failed/partial run without inspecting its artifacts.

Unchanged-trainer control, first200 updates:

```bash
test ! -e "$CTRL1" && uv run --locked isaaclab train "${COMMON[@]}" --checkpoint "$BASE" --run_timestamp "$TAG" --run_name control_b1 "${OVERRIDES[@]}"
```

Data-first trainer, first200 updates:

```bash
test ! -e "$DATA1" && uv run --locked isaaclab train "${COMMON[@]}" --agent rsl_rl_data_first_cfg_entry_point --checkpoint "$BASE" --run_timestamp "$TAG" --run_name data_b1 "${OVERRIDES[@]}"
```

After screening those endpoints, finish each arm's remaining200 updates:

```bash
test ! -e "$CTRL2" && uv run --locked isaaclab train "${COMMON[@]}" --checkpoint "$CTRL1/model_2399.pt" --run_timestamp "$TAG" --run_name control_b2 "${OVERRIDES[@]}"
test ! -e "$DATA2" && uv run --locked isaaclab train "${COMMON[@]}" --agent rsl_rl_data_first_cfg_entry_point --checkpoint "$DATA1/model_2399.pt" --run_timestamp "$TAG" --run_name data_b2 "${OVERRIDES[@]}"
```

The current runner repeats the loaded checkpoint's iteration label as the first
new iteration: endpoints are2399 after200 updates and2598 after400, while the
data-first relative counters are200 and400. Intermediate saves occur every100.
The second block restores the replay and assistance schedule; do not reset it.

## Screen checkpoints at0,200 and400 updates

The initial2200 screen is recorded above. Set `CKPT` and a fresh `SCREEN` for
each arm's200/400 endpoint, then run both commands. Always use the default play
agent: it loads the student and ignores collection assistance/replay.

```bash
CKPT="$DATA1/model_2399.pt"
SCREEN="logs/camera_${TAG}_data_200"
uv run --locked python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint "$CKPT" --policy camera --population nominal --stage F --seeds 42 --batch-size 256 --episodes 256 --output-dir "${SCREEN}_nominal"
uv run --locked python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint "$CKPT" --policy camera --population stage --stage F --seeds 45 --batch-size 256 --episodes 256 --output-dir "${SCREEN}_wide"
```

Repeat with `CTRL1/model_2399.pt`, `CTRL2/model_2598.pt` and
`DATA2/model_2598.pt`, each with its own `SCREEN`. Pick the best nominal-eligible
endpoint per arm by wide success, breaking ties by nominal success. Compare
against2200: nominal may drop at most3 percentage points.

Audit exit2 means its stricter standalone acceptance/coverage gate was not met;
the saved first-episode results are still usable if all jobs completed and
validated. A failed simulation or missing result is not a completed comparison.

## Final paired evaluation after both arms finish

For the baseline and the best endpoint per arm, set `CKPT`/`FINAL` to distinct
values and run the following. Use1024 envs in one batch for exact matched poses.
Final seeds52/49 differ from screen seeds42/45.

```bash
CKPT="$BASE"
FINAL="logs/camera_${TAG}_final_baseline"
uv run --locked python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint "$CKPT" --policy camera --population nominal --stage F --seeds 52 --batch-size 1024 --episodes 1024 --output-dir "${FINAL}_nominal"
uv run --locked python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint "$CKPT" --policy camera --population stage --stage F --seeds 49 --batch-size 1024 --episodes 1024 --output-dir "${FINAL}_wide"
```

Repeat for the selected control and data-first checkpoints, using respective
`FINAL="logs/camera_${TAG}_final_control"` and
`FINAL="logs/camera_${TAG}_final_data"`. Then compare:

```bash
uv run --locked python -m isaaclab_tutorial.tasks.place_box.camera_comparison --baseline-nominal "logs/camera_${TAG}_final_baseline_nominal" --baseline-wide "logs/camera_${TAG}_final_baseline_wide" --candidate control "logs/camera_${TAG}_final_control_nominal" "logs/camera_${TAG}_final_control_wide" --candidate data-first "logs/camera_${TAG}_final_data_nominal" "logs/camera_${TAG}_final_data_wide" --output "logs/camera_${TAG}_final_comparison.json"
```

The comparison validates completed audits, a single checkpoint hash for each
nominal/wide pair, exact commands aside from checkpoint, episode identities and
exact starting poses. It reports paired successes and
descriptive95% bootstrap intervals (10000 resamples,seed0). Adoption requires
at least10 points more wide success and at most3 points nominal drop versus
the common baseline. If both qualify, prefer control unless data-first's
head-to-head wide interval is entirely above0. If neither qualifies, retain
the baseline and stop; do not automatically extend training.

## Training metrics

Read `Loss/student_success_rate` together with `Loss/student_episodes` and
`Loss/student_successes`. A zero rate with zero completed student episodes
contains no student accuracy evidence, especially during teacher-only seeding.
Teacher episodes/successes are logged separately. These per-update training
counts depend on visited states and assistance; independent student-only audits
remain the score for model selection.

`Loss/teacher_control_fraction` reports actual controlled steps, which can lag
the scheduled probability because controller assignments last whole episodes.
`Loss/fresh_phase_*`, `Loss/replay_phase_*` and `Loss/replay_occupancy_*` show
coverage and retention. Compare `Loss/fresh_behavior` and `Loss/replay_behavior`
for fit/forgetting; neither loss has a universal0.1 success threshold.
