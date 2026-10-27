# Local PC setup

Current task state: [HANDOFF.md](HANDOFF.md). The project already passed GPU
state/camera stepping and policy playback here. Keep the working runtime;
reinstallation is not the next task.

## Existing computer

| Component | Verified value |
| --- | --- |
| Runtime | `D:\work\robotics\IsaacLabTutorial`, WSL `/mnt/d/work/robotics/IsaacLabTutorial` |
| Git publication checkout | `D:\work\robotics\issaclab_robotics`, WSL `/mnt/d/work/robotics/issaclab_robotics` |
| WSL distro | `Ubuntu`, Ubuntu24.04 |
| GPU / driver | RTX5070,12 GB /616.92 |
| Python / uv |3.12.3 /0.12.1 |
| Isaac Lab |3.0.0, commit `d061bf35a58273ef0cc8aceae76392166f9b7d15` |
| Torch / torchvision |2.11.0+cu130 /0.26.0+cu130 |
| Newton / Warp / RSL-RL |1.6.1 /1.17.0 /5.5.1 |

The runtime `.venv` is a repaired Linux environment: use WSL, not native Windows
Python. It is not copied into the publication checkout. The old source-machine
`E:` paths, `Ubuntu-24.04` distro name and RTX5060 Ti are obsolete. Installed
optimizer-resume CLI files match the pinned upstream code; those options do
not depend on an uncommitted dependency patch.

From PowerShell, enter the existing runtime:

```powershell
wsl -d Ubuntu --cd /mnt/d/work/robotics/IsaacLabTutorial
```

For visible WSLg playback:

```bash
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
```

CUDA physics uses `--device cuda:0` independently. Without the graphics settings,
OpenGL may select CPU rendering. Do not add `--deterministic` to playback on this pin.

## Fresh clone or another machine

Use Linux/WSL with Python3.12, Git LFS, uv, an NVIDIA GPU and a CUDA13-compatible
driver. GitHub authentication is required for the private repository.

```bash
git clone https://github.com/blueocvn/issaclab_robotics.git
cd issaclab_robotics
git lfs install
git lfs pull
uv sync --locked
uv run pytest -q
```

Git LFS materializes models/assets; pointer text is not a usable model.
[MODEL_INDEX.md](../MODEL_INDEX.md) and
[checkpoint_manifest.json](../checkpoint_manifest.json) identify models/hashes.
Use repository-relative commands in a fresh clone; hardcoded PC paths in the
handoff describe the existing runtime.

## Checks

Run from the selected repository's locked environment:

```bash
uv run --locked isaaclab zero_agent --task IsaacTutorial-Pick-Box-SO101 --num_envs 8 --max_steps 100 --device cuda:0 --visualizer none presets=newton_mjwarp
uv run --locked python -m isaaclab_tutorial.tasks.place_box.check --camera --steps 10 --device cuda:0 --visualizer none
uv run --locked isaaclab zero_agent --task IsaacTutorial-Place-Vial-SO101 --num_envs 8 --max_steps 100 --device cuda:0 --visualizer none presets=newton_mjwarp
uv run pytest -q
```

Most recent full suite:174 passed,2 optional skips for `cv2` and `leapp`.
Those extras are not required for headless PPO/camera simulation. Runtime repair
and GPU check evidence remain in `logs/`. Reading the handoff requires no new
simulation, long training or hardware operation.
