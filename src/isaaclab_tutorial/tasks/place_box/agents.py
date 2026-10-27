"""Separate box checkpoints; the distillation teacher mirrors the box state actor."""

from isaaclab.utils import configclass

from isaaclab_tutorial.tasks.place_vial.config.so101.agents.rsl_rl_distillation_cfg import (
    SO101CameraDistillationRunnerCfg,
)
from isaaclab_tutorial.tasks.place_vial.config.so101.agents.rsl_rl_ppo_cfg import SO101StatePPORunnerCfg


@configclass
class BoxPPORunnerCfg(SO101StatePPORunnerCfg):
    experiment_name = "so101_box_state"
    max_iterations = 800


@configclass
class BoxDistillationRunnerCfg(SO101CameraDistillationRunnerCfg):
    experiment_name = "so101_box_camera_distillation"
    max_iterations = 1600
