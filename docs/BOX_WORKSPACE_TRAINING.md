# Box workspace evaluation and training

Current camera experiment: [CAMERA_DATA_FIRST.md](CAMERA_DATA_FIRST.md) contains
the completed800-update checkpoint screen and direct user-run commands for the
opt-in data-first trainer versus the unchanged-trainer control. Use2200 as the
common experimental baseline; keep all historical results below as provenance.

Start with [HANDOFF.md](HANDOFF.md). Teacher5381 finished training and failed
independent acceptance: grid2,891/3,000; random3,282/3,328. Nominal3,072/3,072
and historical case5273/3 passed. See the
[evaluation report](../logs/workspace_5381_acceptance_20261027_122310/REPORT.md).
The user accepted these observed teacher results as sufficient and requested
camera distillation from5381. That decision supersedes the stricter teacher
promotion gate for this step; original audit thresholds/results remain recorded.
Use a fresh camera student with matching nominal physics. Do not repeat Stage A.
The user launches further long training manually. Commands below reproduce audits.

## Evaluate the latest refinement

On this PC, use the existing runtime in WSL distro `Ubuntu-24.04`:

```bash
cd /mnt/e/issaclab_robotics
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

The user accepted teacher5381 at the observed96.37% grid /98.62% random scores.
Initialize distillation from that PPO checkpoint. Camera1599 embeds its old teacher;
loading it does not replace that teacher. This refers to the archived nominal
camera1599; the new2026-10-27 workspace camera1599 embeds5381 exactly.
Provide direct commands for future runs rather than creating per-run launch scripts.

```bash
uv run --locked isaaclab train --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101-Camera --num_envs 128 --max_iterations 1600 --seed 42 --run_name so101-box-workspace-camera --device cuda:0 --checkpoint "$CKPT" --visualizer none presets=newton_mjwarp,newton_renderer env.events.reset_home.params.position_noise=0.1 env.events.reset_home.params.yaw_noise=3.141592653589793 env.events.reset_home.params.nominal_fraction=0.2
```

The1,600-iteration run above completed. Its final workspace camera1599 scored
455/3,000 grid (15.17%),673/3,328 random (20.22%),2,152/3,072 nominal (70.05%)
and3/3 case527. All starts matched the teacher comparison exactly. See the
[camera evaluation report](../logs/camera_1599_evaluation_20261027_132245/REPORT.md).
Further training and edge-case inspection are deferred by the user. Audit any
actual student with `--policy camera` on the same four populations. For hardware
transfer, first adapt/evaluate the teacher with `sim2real`, then match physics,
renderer and transfer presets for the student. Current variation covers solid
colors and image/camera parameters; printed patterns/logos need coverage and
real-image validation. See [BOX_PICK_PLACE.md](BOX_PICK_PLACE.md).

Preparation verified on2026-10-27: five iterations at128 environments completed,
student parameters updated with finite losses, and the embedded frozen teacher
matched5381 exactly. Evidence: `logs/camera_5381_smoke_verification.json`.
The smoke checkpoint is not a trained student; start the full run from5381.

The subsequent400-update lower-rate experiment retained camera1798: matched
256-start screens showed90.23% nominal and21.09% wide success. Later1900/1997
regressed. Ordinary camera/data diagnostics found limited initial view coverage
(estimated72.4% of full-workspace box centers inside the policy crop) and sparse
post-lift phase visitation in student interior rollouts (5.6% of control steps,
versus30.0% for the teacher). These are diagnostic findings with limitations, not
confirmed causes. See the
[diagnostic report](../logs/camera_visibility_diagnostic_20261027_153417/REPORT.md)
before planning further training. No framing or replay changes have been implemented.

## Optional overnight controller

`workspace_pipeline.py` and root `run_workspace_overnight.ps1` remain in the
codebase. They run A–F with bounded retries, and the Windows wrapper requests
shutdown after success or unresolved failure. The previous job has finished;
this is not the current manual continuation workflow. The wrapper targets the
original PC paths; inspect those paths and power behavior before choosing it
elsewhere. Do not restart it automatically.
