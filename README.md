# SO-101 box and vial robotics

This repository documents the work on an SO-101 robot arm: building a box-to-A4-paper
simulation task, training state and wrist-camera policies, testing wider pickup
positions, and recording what worked and what did not. It includes source code,
trained checkpoints, experiment reports and commands to reproduce the work.

The project extends [IsaacLabTutorial](https://github.com/isaac-sim/IsaacLabTutorial).
The original vial-placement task, license and attribution are retained. The box
task and camera experiments below describe the additional work in this repository.

Current work is on the `camera-data-first` branch. Start with the
[latest results](docs/CAMERA_PPO_RESULTS.md), [handoff](docs/HANDOFF.md) or
[model index](MODEL_INDEX.md) for detailed evidence and checkpoint roles.

## Demo video

[Watch or download the demo (MP4)](media/demo.mp4).

[![SO-101 original vial-placement simulation demo](media/demo.gif)](media/demo.mp4)

This existing video shows the original **vial-placement simulation**. A recorded
box-to-A4 demo is not included yet. To view the trained box camera policy live,
use the playback command below. The video is an illustration, not evidence for
the box evaluation scores.

## Work completed

| Work | What was implemented or tested | Record |
| --- | --- | --- |
| Box-to-A4 task | Upright pickup, physical grasp/lift checks, placement fully inside an A4 marker, release and stability checks | [Task and success criteria](docs/BOX_PICK_PLACE.md) |
| State PPO teacher | Nominal training and wider position/yaw training; grid, random and regression audits | [Workspace training](docs/BOX_WORKSPACE_TRAINING.md), [teacher results](logs/workspace_5381_acceptance_20261027_122310/REPORT.md) |
| Wrist-camera distillation | Teacher-to-student training using wrist RGB and joint feedback; independent workspace evaluation | [Student results](logs/camera_1599_evaluation_20261027_132245/REPORT.md) |
| Data-first and replay experiments | Teacher assistance, phase-balanced replay, checkpoint resume and protected-reference experiments; candidates did not achieve the required Wide gain | [Data-first protocol/results](docs/CAMERA_DATA_FIRST.md), [protected reference](docs/CAMERA_PROTECTED_REFERENCE.md) |
| Native Camera PPO | Verified transfer of the distilled actor into PPO, followed by 30-second and 60-second continuations | [Implementation](docs/CAMERA_PPO.md), [latest results](docs/CAMERA_PPO_RESULTS.md) |
| Published artifacts | Training checkpoints and exported policies in Git LFS, hashes, relocation records and raw evaluation evidence | [Model index](MODEL_INDEX.md), [checkpoint inventory](checkpoint_manifest.json), [path mapping](checkpoint_relocations.json) |

## Latest results

The latest Camera PPO checkpoint is
[`model_1547.pt`](checkpoints/camera_ppo/60s_500/model_1547.pt).
It completed 500 additional iterations from model 1048 with a 60-second episode
limit. Both rows below were evaluated with matching 60-second limits and starts:

| Camera PPO checkpoint | Nominal success | Wide success |
| --- | ---: | ---: |
| Source: model 1048 | 253/256 (98.83%) | 101/256 (39.45%) |
| After 500 additional iterations: model 1547 | 254/256 (99.22%) | 101/256 (39.45%) |

The continuation produced **no measured Wide gain**. These are 256-start selection
screens; no independent 1024-start confirmation or box hardware deployment was
performed. Nominal starts vary by ±1 cm and ±0.1 rad; Wide starts cover ±10 cm and
full upright yaw. See the [comparison and raw evidence](docs/CAMERA_PPO_RESULTS.md).

Earlier milestones use different evaluation protocols and should be read separately:

- The expanded-workspace state teacher scored 2,891/3,000 grid and 3,282/3,328
  random successes. It was retained for distillation despite missing the original
  strict acceptance gates.
- The first expanded-workspace camera student scored 455/3,000 grid and
  673/3,328 random successes, showing a large gap from its teacher.
- Subsequent distillation, replay and PPO experiments are preserved with their
  limitations in the linked reports, including unsuccessful runs.

The latest recorded full verification was 227 tests passed and 2 optional
dependency skips; see [the validation record](docs/CAMERA_PPO_RESULTS.md).
This records the completed run, rather than a fresh test of every new machine.

## Setup

Use Linux x86-64, or Ubuntu 24.04 in WSL2 on Windows. The existing runtime uses
Linux Python inside WSL. Run the commands below in a Linux/WSL terminal from the
repository root.

Prerequisites: Python 3.12, Git, Git LFS, `uv`, an NVIDIA GPU and a CUDA 13-compatible
driver. The locked environment pins Isaac Lab 3.0.0, Torch 2.11.0 and torchvision
0.26.0. Check that `nvidia-smi`, `git lfs version` and `uv --version` work before
installing the project. Authentication is needed if GitHub requests access.

### Fresh clone

```bash
git lfs install
git clone --branch camera-data-first https://github.com/blueocvn/issaclab_robotics.git
cd issaclab_robotics
git lfs pull
uv sync --locked --group dev
uv run --locked pytest -q
```

Git LFS downloads the model and media contents; a text pointer is not a usable
checkpoint or video. Published models are under `checkpoints/`; reports and raw
evaluation evidence are under `logs/`. New runs also write to `logs/` locally.

### Check the simulation and camera

These short GPU checks step the box task and inspect its camera without training:

```bash
uv run --locked isaaclab zero_agent --task IsaacTutorial-Pick-Box-SO101 --num_envs 8 --max_steps 100 --device cuda:0 --visualizer none presets=newton_mjwarp
uv run --locked python -m isaaclab_tutorial.tasks.place_box.check --camera --steps 10 --device cuda:0 --visualizer none
```

For an existing environment without `uv`, use `.venv/bin/python` and
`.venv/bin/isaaclab` directly. For example, replace `uv run --locked pytest` with
`.venv/bin/python -m pytest`. See [local setup notes](docs/LOCAL_PC_SETUP.md) for
machine-specific history; use your own clone path and WSL distro name.

## Run the trained box policy

After setup, this opens a live simulation of the latest Camera PPO policy at
nominal starts, with the 60-second episode limit used for its recorded results:

```bash
uv run --locked isaaclab play --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101-Camera --agent rsl_rl_camera_ppo_cfg_entry_point --checkpoint checkpoints/camera_ppo/60s_500/model_1547.pt --num_envs 4 --device cuda:0 --visualizer newton presets=newton_mjwarp,newton_renderer env.episode_length_s=60.0
```

Press Ctrl+C to stop. The explicit Camera PPO agent is required: the camera task
otherwise defaults to distillation. Playback uses deterministic policy actions;
avoid `--deterministic`, which also changes physics and has a known buffer issue
on this pin.

For WSLg playback, set these before launching if OpenGL selects CPU rendering:

```bash
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
```

To reproduce the 256-start nominal selection protocol without opening a viewer:

```bash
uv run --locked python -m isaaclab_tutorial.tasks.place_box.workspace_audit --checkpoint checkpoints/camera_ppo/60s_500/model_1547.pt --policy camera --agent rsl_rl_camera_ppo_cfg_entry_point --population nominal --stage F --seeds 42 --batch-size 256 --episodes 256 --episode-length-s 60
```

Each audit creates a new output directory under `logs/`. Inspect its manifest and
summary; a completed selection screen can still fail the stricter acceptance
criteria. For Wide playback and the matched comparison, see
[Camera PPO results](docs/CAMERA_PPO_RESULTS.md). State-teacher playback and full
workspace audits are in [the box guide](docs/BOX_PICK_PLACE.md) and
[workspace guide](docs/BOX_WORKSPACE_TRAINING.md).

### Optional Recording and Experiment Tracking

Add `--extra video` to `uv run` commands that record videos with `--video` or
`record_video`. This installs Isaac Lab's MoviePy dependency through `isaaclab[video]`.

For Weights & Biases (W&B) experiment tracking, sign in with:

```bash
uv run --extra wandb wandb login
```

Then add `--extra wandb` to the training command and set `agent.logger=wandb` and
`agent.wandb_project=YOUR_PROJECT`. The tutorial's `wandb` extra forwards to `isaaclab[wandb]`
and temporarily pins a compatible W&B version in [pyproject.toml](pyproject.toml), because
the Isaac Lab logger uses the legacy `wandb.util.generate_id` API. Both extras can be
combined as `uv run --extra video --extra wandb` when recording and tracking a run.

## Tasks

The additional **upright box-to-A4-paper task** is documented in
[docs/BOX_PICK_PLACE.md](docs/BOX_PICK_PLACE.md), including GPU visualization,
physical grasp checks, PPO, exact evaluation and camera distillation.

| Task | Inputs | Training |
| --- | --- | --- |
| `IsaacTutorial-Place-Vial-SO101` | Full state | PPO |
| `IsaacTutorial-Place-Vial-SO101-Camera` | Wrist RGB (2 frames, 64×48) + proprioception | Distillation from a state teacher, or PPO from scratch |
| `IsaacTutorial-Pick-Box-SO101` | 54-dimensional state | PPO, including wider-position/yaw stages |
| `IsaacTutorial-Pick-Box-SO101-Camera` | Wrist RGB + proprioception | Distillation by default; native Camera PPO with an explicit agent |

Add `presets=sim2real` to either task to enable robot-transfer randomization. The physical randomization is
identical in both tasks, so a teacher and its student train under the same physics. See
[docs/SIM2REAL.md](docs/SIM2REAL.md).

## Original vial tutorial

The following training, evaluation and design notes describe the inherited vial
task. For the box work, use [workspace training](docs/BOX_WORKSPACE_TRAINING.md),
[camera experiments](docs/CAMERA_DATA_FIRST.md) and [Camera PPO](docs/CAMERA_PPO.md).

<details>
<summary>Vial training, evaluation and task design</summary>

### Train

```bash
# State teacher
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101 --num_envs 4096 --visualizer none presets=newton_mjwarp

# Camera student, distilled from the teacher checkpoint
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera --num_envs 1024 --checkpoint /abs/path/to/state/model_799.pt --visualizer none presets=newton_mjwarp,newton_renderer
```

During distillation the student acts and the frozen teacher labels every state it visits. The labels are
clipped to the executed `[-1, 1]` action range. Checkpoints go to `logs/rsl_rl/`.

The original vial camera policy can also be trained from scratch with PPO,
without a teacher. The actor sees wrist images and proprioception; the training
critic sees privileged state. This command is separate from the box Camera PPO
experiment documented above.

```bash
uv run isaaclab train --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Camera --agent rsl_rl_ppo_cfg_entry_point --num_envs 1024 --visualizer none presets=newton_mjwarp,newton_renderer
```

### Evaluate

Play starts every episode from the home pose. The exact-evaluation callback counts one episode per
environment and prints a `SO101_EVAL_RESULT` JSON line. Set `SO101_EVALUATION_OUTPUT=path.json` to also save
per-episode outcomes.

```bash
uv run isaaclab play --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101 --num_envs 1024 --checkpoint /abs/path/to/model.pt --visualizer none --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter presets=newton_mjwarp
```

To evaluate a camera policy, use the `-Camera` task with `presets=newton_mjwarp,newton_renderer`, and add
`--agent rsl_rl_ppo_cfg_entry_point` for one trained from scratch.

Do not pass `--deterministic`. On this pin it also enables Warp-wide deterministic physics, which overflows a
collision-kernel buffer in this scene. Playback already uses deterministic policy actions.

### Task design

**Control and actions**

- The policy runs at 30 Hz, and physics runs at 120 Hz.
- Actions are Isaac Lab relative joint-position targets: at every physics step the target is the
  *measured* joint position plus the scaled action, which matches the real servo interface. The scale is
  0.033 rad for the arm joints and 0.02 rad for the gripper.
- A small penalty discourages pushing against the soft joint limits.

**Rewards and success**

- Three physical milestones (grasp, lift, insertion) each pay once per episode.
- Dense shaping guides the approach to the cap and the held vial toward the nearest hole.
- Success means the vial is seated upright, released and almost still for ten consecutive control steps.

**Resets**

- Episodes start from a physics-validated reset dataset (`assets/reset_poses.pt`). It covers eight phases of
  the task, from the home pose through to release.
- Training samples all phases; evaluation starts from home.

To regenerate or inspect the reset dataset:

```bash
uv run generate-so101-resets --output outputs/reset_poses.pt --visualizer none presets=newton_mjwarp
uv run view-so101-resets --dataset outputs/reset_poses.pt --visualizer newton presets=newton_mjwarp
```

</details>

## Layout

```
src/isaaclab_tutorial/
  tasks/place_box/                                box task, camera trainers and workspace audits
  tasks/place_vial/config/so101/env_cfg.py          state task and the sim2real preset
  tasks/place_vial/config/so101/camera_env_cfg.py   wrist-camera task (inherits the state task)
  tasks/place_vial/config/so101/agents/             RSL-RL PPO and distillation configs
  tasks/place_vial/mdp/                             observations, rewards, terminations, events
  tasks/place_vial/reset/                           reset dataset format and generator
  utils/                                            evaluation, LEAPP export, real-robot scripts
checkpoints/                                       published training models and exported policies (Git LFS)
logs/                                              published run evidence and local output from new runs
media/                                             original vial demo GIF and MP4 (Git LFS)
docs/                                              setup, protocols, results and handoff
```

## Documentation map

| Read this | For |
| --- | --- |
| [Handoff](docs/HANDOFF.md) | Current state, checkpoint roles and remaining work |
| [Latest Camera PPO results](docs/CAMERA_PPO_RESULTS.md) | Matched comparison, limitations and evidence |
| [Box task](docs/BOX_PICK_PLACE.md) | Scene, physical success criteria and runtime checks |
| [Workspace training](docs/BOX_WORKSPACE_TRAINING.md) | State/camera training and independent evaluation protocols |
| [Data-first experiments](docs/CAMERA_DATA_FIRST.md) | Completed control/experimental protocol and results |
| [Protected reference](docs/CAMERA_PROTECTED_REFERENCE.md) | Reference collection, replay and preservation experiments |
| [Camera PPO](docs/CAMERA_PPO.md) | Actor transfer, native PPO checkpoints and audit commands |
| [Model index](MODEL_INDEX.md) | Published checkpoints, hashes and validation status |
| [Sim-to-real](docs/SIM2REAL.md) | Original vial calibration/transfer reference; box transfer still needs validation |

Experiment reports retain their original run paths as provenance. Resolve moved
checkpoint paths through [checkpoint_relocations.json](checkpoint_relocations.json).
Completed plans and older handoffs remain in `docs/` or Git history; raw results
remain in the published `logs/` records.

PhysX and OVRTX are available for sim-to-sim checks: install them with `uv sync --extra ovphysx --extra ovrtx`
and run with `presets=physx,ovrtx`. The tutorial is not tuned for those backends.
