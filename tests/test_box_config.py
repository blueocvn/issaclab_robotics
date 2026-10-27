"""Contracts for the separate box tasks and their train/play configurations."""

import gymnasium as gym
import pytest

from isaaclab_tasks.utils.hydra import resolve_presets

import isaaclab_tutorial.tasks  # noqa: F401
from isaaclab_tutorial.tasks.place_box.env_cfg import BoxCameraEnvCfg, BoxEnvCfg


def test_box_tasks_keep_vial_tasks_and_use_separate_agents():
    for suffix in ("", "-Camera"):
        assert gym.spec("IsaacTutorial-Place-Vial-SO101" + suffix)
        spec = gym.spec("IsaacTutorial-Pick-Box-SO101" + suffix)
        assert "place_box.agents" in spec.kwargs["rsl_rl_cfg_entry_point"]


def test_box_scene_has_correct_dimensions_and_no_rack_or_vial():
    cfg = resolve_presets(BoxEnvCfg(), ["newton_mjwarp"])
    assert cfg.scene.box.spawn.size == (0.13, 0.045, 0.09)
    assert cfg.scene.paper.spawn.size[:2] == (0.297, 0.21)
    assert cfg.scene.paper.spawn.collision_props is None
    assert cfg.scene.vial is None and cfg.scene.rack is None
    assert cfg.scene.box.spawn.mass_props.mass == 0.1
    assert cfg.scene.box.init_state.pos == pytest.approx((0.17, -0.10, 0.0805))
    assert cfg.scene.paper.init_state.pos[:2] == pytest.approx((0.17, 0.10))
    assert cfg.episode_length_s == 30
    assert cfg.events.reset_home.func.__name__ == "reset_home"
    assert cfg.events.reset_home.params["position_noise"] == 0.01
    assert cfg.scene.fixed_jaw_contact.filter_prim_paths_expr == ["{ENV_REGEX_NS}/Box"]


def test_camera_student_and_teacher_have_identical_physics_and_state_inputs():
    state = resolve_presets(BoxEnvCfg(), ["newton_mjwarp", "sim2real"])
    camera = resolve_presets(BoxCameraEnvCfg(), ["newton_mjwarp", "newton_renderer", "sim2real"])
    assert camera.scene.box.spawn.size == state.scene.box.spawn.size
    assert camera.actions.to_dict() == state.actions.to_dict()
    assert camera.observations.teacher_state.to_dict() == state.observations.policy.to_dict()
    assert camera.scene.wrist_camera.width == 80
    assert camera.scene.wrist_camera.height == 60
    assert camera.observations.wrist_rgb.image.params["history_length"] == 2
    assert camera.scene.num_envs == 128


def test_custom_layout_updates_visual_marker_and_reset_target_consistently():
    cfg = BoxEnvCfg(pickup_xy=(0.20, -0.08), target_xy=(0.25, 0.12), paper_yaw=0.4)
    cfg = resolve_presets(cfg, ["newton_mjwarp"])
    cfg.configure_layout()  # BoxEnv applies CLI/preset overrides before constructing the scene.
    assert cfg.scene.box.init_state.pos[:2] == pytest.approx((0.15, -0.08))
    assert cfg.scene.paper.init_state.pos[:2] == pytest.approx((0.20, 0.12))
    assert cfg.events.reset_home.params["paper_yaw"] == 0.4


def test_camera_play_preserves_audit_env_count(monkeypatch):
    from isaaclab_tutorial.tasks.place_box import evaluation

    monkeypatch.setattr(evaluation, "EXACT_EVALUATION_ACTIVE", True)
    cfg = resolve_presets(BoxCameraEnvCfg(), ["newton_mjwarp", "newton_renderer"])
    cfg.scene.num_envs = 1024
    cfg.play_mode()
    assert cfg.scene.num_envs == 1024


def test_runtime_friction_override_is_applied_after_newton_contact_initialization():
    cfg = resolve_presets(BoxEnvCfg(), ["newton_mjwarp"])
    cfg.box_friction = 0.45
    cfg.configure_layout()
    assert cfg.events.box_material.mode == "startup"
    assert cfg.events.box_material.params["static_friction_range"] == (0.45, 0.45)
    assert cfg.events.box_material.params["dynamic_friction_range"] == (0.45, 0.45)


@pytest.mark.parametrize("field,value", [("box_mass", float("nan")), ("box_friction", -0.1)])
def test_invalid_physical_configuration_is_rejected(field, value):
    cfg = resolve_presets(BoxEnvCfg(), ["newton_mjwarp"])
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        cfg.configure_layout()


@pytest.mark.parametrize("reset_ids", [[0], [0, 1]])
@pytest.mark.parametrize("have_metrics", [False, True])
def test_reset_keeps_terminal_stage_metrics_alongside_new_episode_logs(monkeypatch, reset_ids, have_metrics):
    import torch

    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tutorial.tasks.place_box.env import BoxEnv

    # Reproduce the installed framework's log replacement without launching physics.
    ids = torch.tensor(reset_ids)
    env = BoxEnv.__new__(BoxEnv)
    env._is_closed = True
    metrics = {"Metrics/on_target_rate": 1.0, "Metrics/success_rate": 0.5} if have_metrics else {}
    env.extras = {"log": {**metrics, "stale_episode_value": 99}, "other_extra": "retained"}

    def manager_reset(self, env_ids):
        assert env_ids is ids
        self.extras["log"] = {"Episode_Reward/success": 2.0, "Episode_Termination/success": 0.5}

    monkeypatch.setattr(ManagerBasedRLEnv, "_reset_idx", manager_reset)
    env._reset_idx(ids)
    assert env.extras["log"] == {
        "Episode_Reward/success": 2.0,
        "Episode_Termination/success": 0.5,
        **metrics,
    }
    assert env.extras["other_extra"] == "retained"
