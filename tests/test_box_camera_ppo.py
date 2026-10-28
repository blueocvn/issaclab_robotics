"""Strict transfer into a fresh PPO actor, without distillation training state."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from rsl_rl.models import CNNModel, MLPModel
from tensordict import TensorDict

import isaaclab_tutorial.tasks  # noqa: F401


def setup_models():
    obs = TensorDict(
        {"wrist_rgb": torch.rand(2, 6, 48, 64), "proprioception": torch.rand(2, 24), "critic": torch.rand(2, 54)}, [2]
    )

    def camera(group):
        return CNNModel(
            obs,
            {group: ["wrist_rgb", "proprioception"]},
            group,
            6,
            hidden_dims=[256, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.05,
                "std_type": "log",
                "std_range": (0.01, 0.15),
            },
            cnn_cfg={
                "output_channels": [16, 32, 32],
                "kernel_size": [5, 3, 3],
                "stride": [2, 2, 2],
                "activation": "elu",
            },
        )

    student, actor = camera("student").eval(), camera("actor").eval()
    student.update_normalization(obs)
    critic = MLPModel(obs, {"critic": ["critic"]}, "critic", 1, hidden_dims=[32], obs_normalization=True)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-5)
    runner = SimpleNamespace(
        alg=SimpleNamespace(actor=actor, critic=critic, optimizer=optimizer), current_learning_iteration=0
    )
    return obs, student, runner


def test_opt_in_camera_ppo_keeps_default_distillation():
    from isaaclab_tutorial.tasks.place_box.agents import BoxCameraPPORunnerCfg

    cfg = BoxCameraPPORunnerCfg()
    assert cfg.obs_groups == {"actor": ["wrist_rgb", "proprioception"], "critic": ["critic"]}
    assert cfg.algorithm.class_name == "PPO"
    assert cfg.algorithm.learning_rate == 1e-5
    assert cfg.algorithm.schedule == "fixed"
    assert cfg.init_at_random_ep_len is False
    spec = gym.spec("IsaacTutorial-Pick-Box-SO101-Camera")
    assert spec.kwargs["rsl_rl_camera_ppo_cfg_entry_point"].endswith(":BoxCameraPPORunnerCfg")
    assert spec.kwargs["rsl_rl_cfg_entry_point"].endswith(":BoxDistillationRunnerCfg")


def test_transfer_preserves_actual_cnn_predictions_normalizer_and_fresh_critic():
    from isaaclab_tutorial.tasks.place_box.camera_ppo import initialize_from_student

    obs, student, runner = setup_models()
    source = {k: v.clone() for k, v in student.state_dict().items()}
    critic = {k: v.clone() for k, v in runner.alg.critic.state_dict().items()}
    initialize_from_student(runner, {"student_state_dict": source})
    assert torch.equal(student(obs), runner.alg.actor(obs))
    assert all(torch.equal(v, source[k]) for k, v in runner.alg.actor.state_dict().items())
    assert all(torch.equal(v, critic[k]) for k, v in runner.alg.critic.state_dict().items())
    assert not runner.alg.optimizer.state
    with torch.no_grad():
        next(runner.alg.actor.parameters()).add_(1)
    assert all(torch.equal(v, source[k]) for k, v in student.state_dict().items())


@pytest.mark.parametrize("bad", ["missing", "shape", "nan"])
def test_bad_transfer_fails_before_actor_mutation(bad):
    from isaaclab_tutorial.tasks.place_box.camera_ppo import initialize_from_student

    _, student, runner = setup_models()
    source = {k: v.clone() for k, v in student.state_dict().items()}
    key = next(k for k, v in source.items() if v.is_floating_point() and v.numel() > 1)
    if bad == "missing":
        source.pop(key)
    if bad == "shape":
        source[key] = torch.zeros(1)
    if bad == "nan":
        source[key].flatten()[0] = float("nan")
    before = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    with pytest.raises(ValueError):
        initialize_from_student(runner, {"student_state_dict": source})
    assert all(torch.equal(v, before[k]) for k, v in runner.alg.actor.state_dict().items())


def test_used_optimizer_is_rejected():
    from isaaclab_tutorial.tasks.place_box.camera_ppo import initialize_from_student

    _, student, runner = setup_models()
    parameter = next(runner.alg.actor.parameters())
    parameter.sum().backward()
    runner.alg.optimizer.step()
    with pytest.raises(ValueError, match="fresh"):
        initialize_from_student(runner, {"student_state_dict": student.state_dict()})


def test_audit_agent_is_opt_in_and_does_not_change_resets():
    from isaaclab_tutorial.tasks.place_box.workspace_audit import build_jobs

    kwargs = dict(
        checkpoint="model.pt",
        policy="camera",
        population="nominal",
        stage="F",
        seeds=[42],
        batch_size=256,
        episodes=256,
    )
    normal = build_jobs(**kwargs)[0]["command"]
    custom = build_jobs(**kwargs, agent="rsl_rl_camera_ppo_cfg_entry_point")[0]["command"]
    assert custom[-2:] == ["--agent", "rsl_rl_camera_ppo_cfg_entry_point"]
    assert custom[:-2] == normal


def test_native_runner_construction_uses_installed_config_migration():
    from isaaclab_tutorial.tasks.place_box.agents import BoxCameraPPORunnerCfg
    from isaaclab_tutorial.tasks.place_box.camera_ppo import create_camera_ppo_runner, initialize_from_student

    obs, student, _ = setup_models()
    env = SimpleNamespace(get_observations=lambda: obs, num_actions=6, num_envs=2, cfg={})
    cfg = BoxCameraPPORunnerCfg()
    cfg.device = "cpu"
    runner = create_camera_ppo_runner(env, cfg)
    initialize_from_student(runner, {"student_state_dict": student.state_dict()})
    assert torch.equal(runner.get_inference_policy()(obs), student(obs))
    assert not runner.alg.optimizer.state
