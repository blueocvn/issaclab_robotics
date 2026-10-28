# Camera PPO fine-tuning for the box task

Camera PPO is standard RSL-RL PPO with a wrist-image/proprioception actor and a
privileged simulator-state critic. This opt-in path does not use teacher action
labels. Box camera distillation remains the default agent.

## Initialization and bounded training

From the existing WSL runtime, use a new output directory:

```bash
.venv/bin/python -m isaaclab_tutorial.tasks.place_box.camera_ppo --checkpoint checkpoints/2200.pt --output logs/new_camera_ppo_trial --iterations 50 --num-envs 128 --device cuda:0 --visualizer none
```

The initializer copies every student tensor into the PPO actor, including
normalization and Gaussian parameters. It rejects incomplete, nonfinite or
shape/dtype-incompatible tensors before loading. Critic and PPO Adam stay fresh;
distillation optimizer and iteration counters are not reused. Pre-update tensor
identity and exact deterministic action parity are checked on actual simulator
observations. Isaac Lab's version migration and runner factory are used.

Defaults:seed67,64 control steps/iteration,4 epochs/4 minibatches, fixed LR1e-5,
PPO clip0.1, entropy coefficient0,20% nominal/80% full Wide resets, full initial
episodes. Exploration starts from2200's saved distribution (~0.05 std), bounded
to0.01–0.15. Standard PPO updates actor/critic normalization from fresh rollouts.
No camera, CNN, physical reward or physical-success condition is changed.

initial.pt is a native PPO checkpoint saved before learning. Training files are
under training/model_N.pt; labels0–49 describe50 new PPO iterations, not the
original distillation iteration numbering. Native PPO checkpoints contain actor,
critic and PPO optimizer state. They must use the Camera PPO agent for playback.
The CLI currently permits1–50 iterations per fresh bounded experiment.

## Playback and auditing

```bash
.venv/bin/isaaclab play --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101-Camera --agent rsl_rl_camera_ppo_cfg_entry_point --checkpoint logs/new_camera_ppo_trial/training/model_49.pt --num_envs 4 --visualizer newton presets=newton_mjwarp,newton_renderer
```

For a controlled nominal selection screen:

```bash
.venv/bin/python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint logs/new_camera_ppo_trial/training/model_49.pt --policy camera --agent rsl_rl_camera_ppo_cfg_entry_point --population nominal --stage F --seeds 42 --batch-size 256 --episodes 256 --output-dir logs/new_camera_ppo_nominal
```

Use population stage,seed45 for full Wide. Every audit must use a new output
directory. The new optional --agent argument also propagates to supplemental
random-audit jobs; omitted arguments preserve historical default commands.
Audit exit2 can indicate insufficient deployment coverage/acceptance even when
the requested256-start selection job completed; inspect manifest and summary.

Compare initial and trained native checkpoints with identical commands/poses.
Initial policy success should also be checked against recorded2200 evidence.
Select on physical task success, never PPO reward alone. Existing experimental
adoption rule:Wide gain>=10 percentage points,Nominal loss<=3 points; confirm on
independent1024-start cohorts (nominal52,Wide49) if eligible before adopting.
No automatic1000-iteration continuation is configured.

## Current trial

The trial completed without qualifying for adoption:Wide83->87/256,Nominal236->230/256. Retain2200; no continuation is queued. Evidence is in logs/camera_ppo_20261028/. The first launch failed during raw
configuration construction before any update; the native factory migration fix
has a regression test. trial_v2 initialized with zero action error and completed
50 iterations/800 optimizer steps. Both models are finite; actor normalizer count
advanced9023488->9433088, exactly409600 new transitions. Native initial/endpoint
selection screens are recorded separately. See that directory's report for the
final decision; these mechanics checks alone do not establish accuracy gains.

Full tests:222 passed,2 optional dependency skips. Changed-file Ruff passed.
Independent read-only review verified transfer, normalization, observation-buffer
ownership, native checkpoint/playback compatibility and agent propagation.

