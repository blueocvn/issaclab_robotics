#!/usr/bin/env bash
set -euo pipefail
cd /mnt/e/issaclab_robotics
export PATH="$HOME/.local/bin:$PATH"

# Fresh wrist-camera student, frozen PPO teacher 5381, matching nominal physics.
# 20% nominal starts and 80% full workspace (+/-10 cm XY, full upright yaw).
checkpoint=logs/rsl_rl/so101_box_state/2026-10-27_03-19-50_so101-box-workspace-F-refine/model_5381.pt
printf '%s  %s\n' de6789715cdadce05ac77b615c52bb827f02e9964186774fea0e9087e71c4c5d "$checkpoint" | sha256sum --check

uv run --locked isaaclab train \
  --rl_library rsl_rl \
  --task IsaacTutorial-Pick-Box-SO101-Camera \
  --num_envs 128 \
  --max_iterations 1600 \
  --seed 42 \
  --run_name so101-box-workspace-camera-5381 \
  --device cuda:0 \
  --checkpoint "$checkpoint" \
  --visualizer none \
  presets=newton_mjwarp,newton_renderer \
  agent.algorithm.learning_rate=0.0005 \
  env.events.reset_home.params.position_noise=0.1 \
  env.events.reset_home.params.yaw_noise=3.141592653589793 \
  env.events.reset_home.params.nominal_fraction=0.2 \
  2>&1 | tee "logs/camera_5381_training_$(date +%Y%m%d_%H%M%S).log"
