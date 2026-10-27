# Box workspace evaluation and training

Start with [HANDOFF.md](HANDOFF.md). Latest teacher5381 has finished training
but needs independent acceptance. Do not repeat Stage A or the completed
refinement before evaluating it. The user launches further long training manually.

## Evaluate the latest refinement

On this PC, use the existing runtime in WSL distro `Ubuntu`:

```bash
cd /mnt/d/work/robotics/IsaacLabTutorial
CKPT='logs/rsl_rl/so101_box_state/2026-10-27_03-19-50_so101-box-workspace-F-refine/model_5381.pt'
uv run --locked python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint "$CKPT" --policy teacher --population stage --stage F --seeds 43 45 --batch-size 1024 --output-dir logs/workspace_5381_stage_screen_20261027
```

Every run needs a new output directory. `--dry-run` prints commands without
simulation. The audit saves exact first-episode outcomes, commands, checkpoint
SHA256, reset provenance and a summary. Model4982 scored1,009/1,024 on43 and
1,013/1,024 on45; these are comparison starts, not completely unused seeds.
Final groups also include46/47.

Generic stage-report `pass` means≥95% overall. Stage F promotion additionally
requires **≥99% overall and≥95% in every spatial/yaw bin**, followed by all four
final populations. A small stage screen alone is not acceptance. Training
estimates include nominal-mixture starts and stochastic actions; do not compare
them directly with pure wider-distribution deterministic playback scores.

## Final acceptance populations

Keep `CKPT` set to the exact model being accepted; run populations separately:

```bash
uv run --locked python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint "$CKPT" --policy teacher --population grid --stage F --batch-size 1024 --output-dir logs/workspace_5381_teacher_grid
uv run --locked python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint "$CKPT" --policy teacher --population random --stage F --batch-size 1024 --output-dir logs/workspace_5381_teacher_random
uv run --locked python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint "$CKPT" --policy teacher --population nominal --batch-size 1024 --output-dir logs/workspace_5381_teacher_nominal
uv run --locked python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint "$CKPT" --policy teacher --population regression --output-dir logs/workspace_5381_teacher_case527
```

| Population | Coverage and gate |
| --- | --- |
| Grid | 5×5 centers, 8 yaw angles45° apart over[-180°,180°), 5 repeats/case in groups45/46/47: 3,000 episodes; ≥99% overall, ≥95% per marginal spatial/yaw bin, ≥100 samples/bin and complete case identities |
| Random | Initially1,024/group45/46/47; predetermined256-start supplements only for insufficient bin counts; same99%/95% gates and three groups |
| Nominal | 1,024/group42/43/44 in the old±1 cm/±0.1 rad range; teacher≥99.9%, camera≥99.5% |
| Regression | Three replays of saved historical case527; all must succeed |

Audits use `nominal_fraction=0.0`. Never pool easy and hard populations. All200
joint grid-case results are reported even though gates use marginal bins.
Repeats measure rollout variation, not new poses. Each batch has a distinct
recorded physics/reset seed; reset draws are independent of actor initialization.
Regression defaults to `logs/box_release_v3_1997_evaluation.json`.

Exit0 means the requested population passes;2 means failure or insufficient
coverage. A partial `--max-jobs` run cannot pass. Execution exceptions are
separate from policy failures and leave logs for diagnosis. Finite testing
does not establish universal reliability or real-robot transfer.

## If another PPO block is needed

Inspect failure starts and grasp/lift/placement flags first. Select the next
checkpoint by independent results, not filename recency. Keep the physical
contract unchanged. Supply the user a command instead of launching long training.
The following is a **template**, not the next required step:

```bash
uv run --locked isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101 --num_envs 512 --max_iterations 400 --seed 42 --run_name so101-box-workspace-F-followup --device cuda:0 --checkpoint "$CKPT" --no-resume_optimizer_state --visualizer none presets=newton_mjwarp agent.algorithm.learning_rate=0.00005 agent.algorithm.schedule=fixed env.events.reset_home.params.position_noise=0.1 env.events.reset_home.params.yaw_noise=3.141592653589793 env.events.reset_home.params.nominal_fraction=0.2
```

The negative optimizer flag restores actor/critic/normalization/iteration with
a fresh optimizer at the specified rate. Omitting it resumes the saved optimizer
and its rate. Iteration numbers continue; select actual numeric checkpoints
within the intended run.

For a new training effort, the implemented curriculum is:

| Stage | Position half-width | Yaw half-width |
| --- | ---: | ---: |
| A |0.02 m |0.1 rad |
| B |0.05 m |0.1 rad |
| C |0.10 m |0.1 rad |
| D |0.10 m |π/4 |
| E |0.10 m |π/2 |
| F |0.10 m |π |

Training uses20% nominal/80% current-stage starts. A–E advancement uses≥95%
screening and failure inspection; F uses the stricter gates above. A–E training
is already complete; this table is not an instruction to restart it.

## Camera and sim-to-real stage

Only after teacher acceptance, initialize distillation from that accepted PPO
checkpoint. Camera1599 embeds its old teacher; loading it does not replace that teacher.

```bash
uv run --locked isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101-Camera --num_envs 128 --max_iterations 1600 --seed 42 --run_name so101-box-workspace-camera --device cuda:0 --checkpoint "$CKPT" --visualizer none presets=newton_mjwarp,newton_renderer env.events.reset_home.params.position_noise=0.1 env.events.reset_home.params.yaw_noise=3.141592653589793 env.events.reset_home.params.nominal_fraction=0.2
```

This is a future user-launched command, not the approved next run. Audit the
actual student with `--policy camera` on the same four populations. For hardware
transfer, first adapt/evaluate the teacher with `sim2real`, then match physics,
renderer and transfer presets for the student. Current variation covers solid
colors and image/camera parameters; printed patterns/logos need coverage and
real-image validation. See [BOX_PICK_PLACE.md](BOX_PICK_PLACE.md).

## Optional overnight controller

`workspace_pipeline.py` and root `run_workspace_overnight.ps1` remain in the
codebase. They run A–F with bounded retries, and the Windows wrapper requests
shutdown after success or unresolved failure. The previous job has finished;
this is not the current manual continuation workflow. The wrapper targets the
original PC paths; inspect those paths and power behavior before choosing it
elsewhere. Do not restart it automatically.
