"""Reset coverage and compatibility for the wider box workspace."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab_tutorial.tasks.place_box import mdp


def helpers():
    from isaaclab_tutorial.tasks.place_box import workspace

    return workspace


def test_stage_limits_and_grid_cover_boundaries_yaw_and_repeats():
    w = helpers()
    assert w.STAGES["A"] == (0.02, 0.1)
    assert w.STAGES["F"] == (0.1, math.pi)
    offsets = w.grid_offsets(torch.arange(400), 0.1, math.pi)
    assert offsets.shape == (400, 3)
    assert torch.equal(offsets[:200], offsets[200:])
    assert len(torch.unique(offsets[:200], dim=0)) == 200
    assert torch.unique(offsets[:, 0]).tolist() == pytest.approx([-0.1, -0.05, 0, 0.05, 0.1])
    assert torch.unique(offsets[:, 1]).tolist() == pytest.approx([-0.1, -0.05, 0, 0.05, 0.1])
    assert torch.unique(offsets[:, 2]).tolist() == pytest.approx([i * math.pi / 4 for i in range(-4, 4)])


def test_nominal_default_preserves_original_rng_sequence():
    w = helpers()
    torch.manual_seed(77)
    xy = torch.empty(12, 2).uniform_(-0.01, 0.01)
    yaw = torch.empty(12).uniform_(-0.1, 0.1)
    torch.manual_seed(77)
    offsets, nominal = w.sample_offsets(12, "cpu", 0.01, 0.1)
    assert torch.equal(offsets[:, :2], xy)
    assert torch.equal(offsets[:, 2], yaw)
    assert not nominal.any()


def test_training_mixture_preserves_nominal_ranges_and_uses_wider_starts():
    offsets, nominal = helpers().sample_offsets(
        10000, "cpu", 0.1, math.pi, nominal_fraction=0.2, generator=torch.Generator().manual_seed(42)
    )
    assert 0.18 < nominal.float().mean() < 0.22
    assert offsets[nominal, :2].abs().max() <= 0.01
    assert offsets[nominal, 2].abs().max() <= 0.1
    assert offsets[:, :2].abs().max() <= 0.1
    assert offsets[:, 2].abs().max() <= math.pi
    assert offsets[~nominal, :2].abs().max() > 0.09


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position_noise": -1},
        {"position_noise": float("nan")},
        {"yaw_noise": float("inf")},
        {"nominal_fraction": 1.1},
        {"nominal_fraction": -0.1},
    ],
)
def test_invalid_reset_distribution_is_rejected(kwargs):
    args = dict(count=2, device="cpu", position_noise=0.1, yaw_noise=math.pi)
    args.update(kwargs)
    with pytest.raises(ValueError):
        helpers().sample_offsets(**args)


class Scene(dict):
    pass


def reset_fixture(monkeypatch, n=12):
    pose = torch.tensor([[0.17, -0.1, 0.0805, 0, 0, 0, 1.0]]).repeat(n, 1)
    written = {}
    robot = SimpleNamespace(data=SimpleNamespace(default_joint_pos=SimpleNamespace(torch=torch.zeros(n, 6))))
    robot.write_joint_position_to_sim_index = lambda **kw: None
    robot.write_joint_velocity_to_sim_index = lambda **kw: None
    robot.set_joint_position_target_index = lambda **kw: None
    robot.set_joint_velocity_target_index = lambda **kw: None
    box = SimpleNamespace(data=SimpleNamespace(default_root_pose=SimpleNamespace(torch=pose)))
    box.write_root_pose_to_sim_index = lambda **kw: written.update(kw)
    box.write_root_velocity_to_sim_index = lambda **kw: None
    scene = Scene(robot=robot, box=box)
    scene.env_origins = torch.zeros(n, 3)
    scene.env_origins[:, 0] = torch.arange(n) * 1.1
    env = SimpleNamespace(scene=scene, num_envs=n, device="cpu")
    monkeypatch.setattr(mdp, "progress", lambda _: SimpleNamespace(reset=lambda ids: None))
    return env, written


def test_partial_grid_reset_changes_only_selected_pose_and_provenance(monkeypatch):
    env, written = reset_fixture(monkeypatch)
    mdp.reset_home(env, None, position_noise=0.1, yaw_noise=math.pi, reset_mode="grid")
    old_pose = env._box_initial_pose.clone()
    old_case = env._box_reset_case.clone()
    mdp.reset_home(
        env, torch.tensor([1, 7]), position_noise=0.05, yaw_noise=0.1, reset_mode="grid", grid_start_index=200
    )
    untouched = torch.tensor([0, 2, 3, 4, 5, 6, 8, 9, 10, 11])
    assert torch.equal(env._box_initial_pose[untouched], old_pose[untouched])
    assert torch.equal(env._box_reset_case[untouched], old_case[untouched])
    assert env._box_reset_case[[1, 7]].tolist() == [201, 207]
    assert written["env_ids"].tolist() == [1, 7]
    assert torch.allclose(written["root_pose"][:, 3:].norm(dim=-1), torch.ones(2))


def test_audit_sampling_seed_is_independent_of_global_rng_and_model_initialization(monkeypatch):
    a, _ = reset_fixture(monkeypatch)
    b, _ = reset_fixture(monkeypatch)
    torch.manual_seed(1)
    mdp.reset_home(a, None, position_noise=0.1, yaw_noise=math.pi, sampling_seed=45)
    torch.manual_seed(999)
    mdp.reset_home(b, None, position_noise=0.1, yaw_noise=math.pi, sampling_seed=45)
    assert torch.equal(a._box_initial_pose, b._box_initial_pose)
    before = a._box_initial_pose.clone()
    mdp.reset_home(a, None, position_noise=0.1, yaw_noise=math.pi, sampling_seed=45)
    assert not torch.equal(a._box_initial_pose, before)


def test_saved_pose_replay_keeps_exact_local_pose(monkeypatch, tmp_path):
    env, _ = reset_fixture(monkeypatch, n=3)
    path = tmp_path / "poses.json"
    pose = [0.175, -0.095, 0.0805, 0, 0, 0.03, math.sqrt(1 - 0.03**2)]
    path.write_text(json.dumps({"poses": [pose]}))
    mdp.reset_home(env, None, reset_mode="replay", replay_path=str(path))
    assert torch.allclose(env._box_initial_pose, torch.tensor([pose]).repeat(3, 1), atol=1e-6)
    assert env._box_reset_source.tolist() == [3, 3, 3]


def test_grid_and_replay_cannot_mix_nominal_training_starts(monkeypatch):
    env, _ = reset_fixture(monkeypatch)
    with pytest.raises(ValueError, match="nominal"):
        mdp.reset_home(env, None, reset_mode="grid", nominal_fraction=0.2)


@pytest.mark.parametrize("mode", ["random", "grid"])
def test_framework_int32_reset_ids_are_compatible_with_provenance_storage(monkeypatch, mode):
    env, _ = reset_fixture(monkeypatch)
    mdp.reset_home(env, torch.arange(env.num_envs, dtype=torch.int32), reset_mode=mode)
    assert env._box_reset_source.dtype == torch.long
    assert env._box_reset_case.dtype == torch.long
