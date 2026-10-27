# Local PC setup and verification — 2026-10-26

The transferred project runs on this PC using the existing WSL Ubuntu 24.04
installation. GPU simulation, wrist-camera rendering, original vial-task
execution and saved box-policy inference were verified before proceeding with
any release-v3 work. No training or task-behavior changes were made during this
initial setup stage. The subsequent handoff review, diagnostic fixes and short
512-environment PPO check are recorded in [RELEASE_V3_REVIEW.md](RELEASE_V3_REVIEW.md).

**Project context**

This is an Isaac Lab 3.0 project for the SO-101 arm. The original task learns to
place a vial upright in any rack hole. The additional box task picks up an
upright 13 x 4.5 x 9 cm box and places it inside a calibrated A4 paper zone.
Both have state-policy and wrist-camera variants. State teachers use RSL-RL
PPO; camera students use teacher distillation. Simulation uses Newton MJWarp.

The task registrations are in `src/isaaclab_tutorial/tasks/`. Box configuration,
geometry, episode progress, rewards, evaluation and diagnostics are in
`src/isaaclab_tutorial/tasks/place_box/`. Local scene assets and the vial reset
dataset are in `src/isaaclab_tutorial/assets/`. Checkpoints are under
`logs/rsl_rl/`.

Read `HANDOFF_BOX_RELEASE_V3.md` as the latest handoff. Lift-v2 learned grasp,
lift and transport but did not learn successful release. Release-v3 reward
changes are present. The handoff's historical 8/8 scripted-release success is
not learned-policy success. The new local checks establish runtime readiness,
not release-v3 convergence or a successful full placement audit.

**Current environment**

| Component | Verified value |
| --- | --- |
| Windows repository | `D:\work\robotics\IsaacLabTutorial` |
| WSL repository | `/mnt/d/work/robotics/IsaacLabTutorial` |
| WSL distribution name | `Ubuntu` (Ubuntu 24.04.2 LTS, WSL 2) |
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB |
| NVIDIA driver | 616.92 |
| Python | 3.12.3, system interpreter `/usr/bin/python3.12` |
| Virtual environment | Project `.venv`, Linux environment |
| uv | 0.12.1, already available through the Ubuntu login-shell PATH |
| Isaac Lab | 3.0.0, pinned Git commit `d061bf35a58273ef0cc8aceae76392166f9b7d15` |
| PyTorch / torchvision | 2.11.0+cu130 / 0.26.0+cu130 |
| Newton / Warp / RSL-RL | 1.6.1 / 1.17.0 / 5.5.1 |
| Git branch / HEAD | `feat/so101-box-placement` / `7efef771fa0bfbcb9683193ff100ce9f015fb17e` |

The old handoff's `E:` paths, `Ubuntu-24.04` distribution name and RTX 5060 Ti
hardware describe the source machine. Use the values above on this PC.
The existing `.venv` is for Linux; run it through WSL.

**Setup performed**

The transferred interpreter links (`.venv/bin/python`, `python3`, `python3.12`)
and `lib64` link had become zero-byte regular files. Recreated those links and
updated 88 launch/activation scripts that referenced the old
`/mnt/e/robotics/IsaacLabTutorial` directory. Updated the editable package path,
then ran `uv sync --locked` to reinstall only the local project package.
The existing dependency installation was retained.

`uv sync --locked --dry-run` subsequently reported **Would make no changes**.
`uv pip check --python .venv/bin/python` reported all 207 installed packages
compatible. A scan found no empty shared libraries in site-packages. CUDA was
available and an actual GPU tensor calculation returned the expected result.
Git LFS is available in WSL and the tracked LFS files contain materialized
content. Existing source changes, assets and checkpoints were retained.

**Verification results**

All commands below were run from the WSL repository directory and exited 0.

| Check | Result | Evidence under `logs/` |
| --- | --- | --- |
| Box state scene, 8 environments, 100 steps, CUDA | Finite observations; 54D policy/critic; resting bottom height about -0.00000313 m | `local_pc_box_smoke.log` |
| Box camera scene, 8 environments, 10 steps, CUDA | Finite observations; wrist RGB shape `[8, 6, 48, 64]`; visible contrast | `local_pc_camera_smoke.log` |
| Original vial CLI task, 8 environments, 100 steps, CUDA | Scene, reset dataset and stepping completed | `local_pc_vial_smoke.log` |
| Lift-v2 exported policy, 8 environments, 100 inference steps | Policy loaded and trace written | `local_pc_policy_smoke.log`, `local_pc_policy_trace.json` |
| Full pytest suite | **127 passed, 2 skipped**, 35 TorchScript deprecation warnings | `local_pc_pytest.log` |
| Ruff on box sources and tests | All checks passed | `local_pc_ruff.log` |
| Locked environment sync | Only local editable project refreshed | `local_pc_uv_sync.log` |

The skipped tests require optional `cv2` (hardware deployment) and `leapp`
(policy export). These extras were already absent from the copied environment.
Headless simulation and camera rendering were verified; an interactive viewer,
hardware deployment, full PPO continuation and a full placement audit were not
part of these local setup checks. A subsequent two-iteration, 512-environment
PPO smoke run passed on this 12 GB GPU; see the handoff review report. The long
continuation remains user-launched.

**Run on this PC**

From PowerShell, open the project in the correct WSL distribution:

```powershell
wsl -d Ubuntu --cd /mnt/d/work/robotics/IsaacLabTutorial
```

Inside that Ubuntu shell:

```bash
uv sync --locked
uv run --locked python -m isaaclab_tutorial.tasks.place_box.check --steps 100 --device cuda:0 --visualizer none
uv run --locked python -m isaaclab_tutorial.tasks.place_box.check --camera --steps 10 --device cuda:0 --visualizer none
uv run --locked isaaclab zero_agent --task IsaacTutorial-Place-Vial-SO101 --num_envs 8 --max_steps 100 --device cuda:0 --visualizer none presets=newton_mjwarp
uv run --locked pytest -q
uv run --locked ruff check src/isaaclab_tutorial/tasks/place_box tests/test_box*
```

A single PowerShell command for the box smoke check:

```powershell
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/work/robotics/IsaacLabTutorial; uv run --locked python -m isaaclab_tutorial.tasks.place_box.check --steps 100 --device cuda:0 --visualizer none'
```

**Continuation boundary**

The lift-v2 continuation checkpoint is present at:

```text
logs/rsl_rl/so101_box_state/2026-10-26_18-39-43_so101-box-lift-v2/model_1598.pt
```

Its `exported/policy.pt` sibling was used for the inference smoke check.
No full training run, scripted release reproduction, release-v3 code changes,
commit, merge, push or hardware operation was performed. The existing untracked
feature files remain in place. Follow the handoff's remaining review and
user-launched training workflow after this setup stage.
