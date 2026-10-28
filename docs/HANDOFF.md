# SO-101 box task handoff

All training and evaluation processes are stopped. Work is on `camera-data-first`.
Models are now consolidated under `checkpoints/` and use Git LFS.

## Current results

The latest Camera PPO checkpoint is`checkpoints/camera_ppo/60s_500/model_1547.pt`.
It completed 500 additional PPO iterations from model 1048 with 60-second episodes.
Matched 256-start screens:Nominal 254/256 (99.22%),Wide 101/256 (39.45%).
The same-horizon source scored 253/256 Nominal and 101/256 Wide:there was no Wide
gain. Full evidence,hashes,limits and playback commands are in
[CAMERA_PPO_RESULTS.md](CAMERA_PPO_RESULTS.md).

At 30 seconds,model 1048 scored 255/256 Nominal (99.61%) and 94/256 Wide (36.72%).
The original camera distillation reference 2200 scored 229/256 Nominal (89.45%)
and 80/256 Wide (31.25%) on its historical selection screens. These different
horizons/runs are not a controlled before/after comparison.

## Checkpoint locations

|Role|Current path|
|---|---|
|Distillation reference|`checkpoints/2200.pt`|
|Camera PPO,30-second1000-iteration continuation|`checkpoints/camera_ppo/30s_1000/model_1048.pt`|
|Latest Camera PPO,60-second500-iteration continuation|`checkpoints/camera_ppo/60s_500/model_1547.pt`|
|Initial PPO trial and exact transferred actor|`checkpoints/camera_ppo/trial_50/`|
|Previously published training models|`checkpoints/archive/`|
|Previously published exported policies|`checkpoints/exports/`|

All Camera PPO saves and the previous published archive are retained. Nineteen
unselected replay-heavy experimental checkpoints are preserved locally under
`checkpoints/local_experiments/` and excluded from publication. Raw collection
bundles and regenerated exports remain local when not already published.
`checkpoint_manifest.json ` inventories published PT files;`checkpoint_relocations.json `
maps every moved original path to its new location and records verified hashes.
Historical run configs/audit JSON retain original paths as provenance.

## Runtime and implementation

Use WSL Ubuntu-24.04,repository `/mnt/e/issaclab_robotics`,existing `.venv`.
Invoke `.venv/bin/python ` or`.venv/bin/isaaclab` directly;uv is unavailable here.
The camera task defaults to distillation. Native Camera PPO requires
`--agent rsl_rl_camera_ppo_cfg_entry_point`. PPO actor uses wrist RGB and
proprioception;only the training critic receives simulator state. PPO checkpoints
contain actor,critic and PPO optimizer state;do not load them as distillation.

The opt-in camera_ppo initializer strictly transfers every 2200 student tensor,
including normalization,into a fresh PPO actor. Critic and Adam start fresh;
actual simulator action parity was exactly 0.0 before the first update. Normal
PPO continuations restore actor/critic/Adam. The installed-version migration and
standard runner factory are required and covered by regression tests.
See [CAMERA_PPO.md](CAMERA_PPO.md)for native checkpoint commands.

For 60-second audits,use `--episode-length-s 60` on workspace_audit for both source
and candidate. For playback,use `env.episode_length_s=60.0`. The default task
horizon is still 30 seconds. Physical success conditions were never relaxed.

## Other completed work

Protected-reference replay,reference collection and resume behavior are
implemented and tested. Teacher-only replay,data-first continuations and
baseline-action preservation experiments did not achieve the agreed Wide gain.
Their reports/scripts/raw audit evidence are preserved under logs;large failed
checkpoint payloads remain local. See [CAMERA_PROTECTED_REFERENCE.md](CAMERA_PROTECTED_REFERENCE.md),
[CAMERA_DATA_FIRST.md](CAMERA_DATA_FIRST.md),and the archived
[handoff history](archive/HANDOFF_2026-10-28_history.md).

Full verification:227 tests passed,2 optional dependency skips,35 upstream
TorchScript warnings;Ruff passed. Source checkpoints and moved files were
hash-verified. No hardware deployment,merge or further training is authorized
by this publication. User requested commit and push of the current branch.
