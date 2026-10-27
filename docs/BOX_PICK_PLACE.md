# SO-101 upright box transfer

Current state: [HANDOFF.md](HANDOFF.md). Commands/gates:
[BOX_WORKSPACE_TRAINING.md](BOX_WORKSPACE_TRAINING.md). Models:
[MODEL_INDEX.md](../MODEL_INDEX.md).

## Tasks and scene

| Task | Policy inputs | Training |
| --- | --- | --- |
| `IsaacTutorial-Pick-Box-SO101` | 54D state | PPO teacher |
| `IsaacTutorial-Pick-Box-SO101-Camera` | Two 64×48 RGB frames plus joint feedback/history | Distillation student |

The original vial tasks remain available. The box is 13 cm long, 4.5 cm thick,
9 cm high; pinch across its thickness. It starts upright on the desk and is
placed inside a 29.7×21 cm white A4 marker. The marker is visual; the desk
supports the box. Nominal mass is a provisional 0.10 kg, friction 0.7.

Coordinates are metres in world-parallel workcell axes translated to robot-base
XY=(-0.05,0). Default pickup A=(0.22,-0.10), paper B=(0.22,+0.10), tabletop
Z=0.035. Subtract 0.05 from workcell X for single-environment scene X.
The box's 13 cm edge and paper's 29.7 cm edge initially align with X.

Overrides `env.pickup_xy`, `env.target_xy`, `env.paper_yaw`, `env.box_mass` and
`env.box_friction` are applied before scene construction so visuals, collisions,
resets and evaluation use the same layout.

## Physical success contract

1. At least two consecutive steps of bilateral contact, then a real lift of
   the box bottom by at least 3 cm.
2. Every projected cuboid corner at least 5 mm inside the A4 boundary.
3. Tilt≤10°, bottom within 3 mm of the tabletop, actual table contact.
4. No sensed jaw contacts; jaw opening>50 mm, pad midpoint>80 mm from box center.
5. Linear speed<0.03 m/s and angular speed<0.3 rad/s for 15 consecutive control
   steps (0.5 seconds).

Placement anywhere entirely inside the paper counts; exact centering/final yaw
are not required. Control is 30 Hz, physics 120 Hz, timeout 30 seconds. Do not
relax success gates to improve a score.

## Reset distributions

Defaults: ±1 cm XY and ±0.1 rad yaw. Wider goal: ±10 cm XY and full upright yaw,
paper fixed. Starting on a side is outside the task.

| `env.events.reset_home.params` | Purpose |
| --- | --- |
| `position_noise`, `yaw_noise` | Position/yaw half-widths in metres/radians |
| `nominal_fraction=0.2` | Training mixture: 20% nominal, 80% current range |
| `nominal_fraction=0.0` | Pure current-distribution audit |
| `reset_mode=grid` | 5×5 centers including boundaries, 8 yaw angles |
| `sampling_seed` | Reset RNG independent of model initialization |
| `reset_mode=replay`, `replay_path` | Explicit saved local-frame poses |

Uniform random starts binned afterward are not a fixed grid. Policy failure
alone does not establish mechanical unreachability.

## Runtime checks

Run from the repository root in its locked Linux environment:

```bash
uv run --locked isaaclab zero_agent --task IsaacTutorial-Pick-Box-SO101 --num_envs 8 --max_steps 100 --device cuda:0 --visualizer none presets=newton_mjwarp
uv run --locked python -m isaaclab_tutorial.tasks.place_box.check --camera --steps 10 --device cuda:0 --visualizer none
uv run --locked python -m isaaclab_tutorial.tasks.place_box.check --grasp-probe --device cuda:0 --visualizer none
```

The grasp probe uses IK and scripted jaw motion for an isolated contact/lift
check. It is not a learned policy or a verified collision-free approach.
Add `--sim2real` to inspect that preset.

## Playback

Example for latest state refinement 5381:

```bash
CKPT='logs/rsl_rl/so101_box_state/2026-10-27_03-19-50_so101-box-workspace-F-refine/model_5381.pt'
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
uv run --locked isaaclab play --rl_library rsl_rl --task IsaacTutorial-Pick-Box-SO101 --num_envs 1 --device cuda:0 --checkpoint "$CKPT" --visualizer newton presets=newton_mjwarp env.events.reset_home.params.position_noise=0.1 env.events.reset_home.params.yaw_noise=3.141592653589793
```

Ctrl+C stops playback. Use the camera task and
`presets=newton_mjwarp,newton_renderer` for a camera checkpoint. Reusing a seed
repeats sampled starts. Do not use `--deterministic`; actions already are
deterministic, and that flag changes physics with a known buffer issue.
Visible playback is not the exact acceptance audit.

## Transfer scope

The box sim-to-real preset varies physical parameters and, for the camera,
solid colors, mount and image formation. Printed logos/pattern textures are
not covered yet. Current trained checkpoints use nominal physics.
Before hardware use, verify box-specific calibration, action scales, appearance
coverage and success under matching transfer presets. The vial-specific
[SIM2REAL.md](SIM2REAL.md) is a reference, not a box deployment recipe.
