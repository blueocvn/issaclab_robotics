"""Separate box checkpoints; the distillation teacher mirrors the box state actor."""

from isaaclab.utils import configclass

from isaaclab_tutorial.tasks.place_vial.config.so101.agents.rsl_rl_distillation_cfg import (
    SO101CameraDistillationRunnerCfg,
)
from isaaclab_tutorial.tasks.place_vial.config.so101.agents.rsl_rl_ppo_cfg import (
    SO101CameraPPORunnerCfg,
    SO101StatePPORunnerCfg,
)


@configclass
class BoxPPORunnerCfg(SO101StatePPORunnerCfg):
    experiment_name = "so101_box_state"
    max_iterations = 800


@configclass
class BoxDistillationRunnerCfg(SO101CameraDistillationRunnerCfg):
    experiment_name = "so101_box_camera_distillation"
    max_iterations = 1600


@configclass
class BoxDataFirstDistillationRunnerCfg(BoxDistillationRunnerCfg):
    """Explicit experimental entry point; the normal box/vial agents stay unchanged."""

    seed = 43
    experiment_name = "so101_box_camera_data_first"
    max_iterations = 200
    init_at_random_ep_len = False

    def __post_init__(self):
        self.algorithm.class_name = "isaaclab_tutorial.tasks.place_box.distillation:DataFirstDistillation"
        self.algorithm.learning_rate = 1.0e-4


@configclass
class BoxAnchoredDistillationRunnerCfg(BoxDataFirstDistillationRunnerCfg):
    """Protected-reference experiment, separate from the original data-first arm."""

    experiment_name = "so101_box_camera_anchored"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "isaaclab_tutorial.tasks.place_box.reference_replay:AnchoredDistillation"
        self.algorithm.reference_path = None


@configclass
class BoxCameraPPORunnerCfg(SO101CameraPPORunnerCfg):
    """Conservative reward-based fine-tuning of the camera2200 actor."""

    seed = 67
    experiment_name = "so101_box_camera_ppo"
    max_iterations = 50
    save_interval = 25
    init_at_random_ep_len = False

    def __post_init__(self):
        self.algorithm.learning_rate = 1e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.clip_param = 0.1
        self.algorithm.entropy_coef = 0.0
        self.algorithm.num_learning_epochs = 4
        self.algorithm.num_mini_batches = 4
        self.actor.distribution_cfg.init_std = 0.05
        self.actor.distribution_cfg.std_range = (0.01, 0.15)
