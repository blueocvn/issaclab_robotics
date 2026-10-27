"""Regressions for a teacher that pressed down and never discovered lifting."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tutorial.tasks.place_box import mdp


def fixture(monkeypatch, n=1):
    position = torch.tensor([[0.17, -0.10, 0.08]]).repeat(n, 1)
    center = position.clone()
    q = dict(
        position=position,
        quaternion=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1),
        paper_position=torch.tensor([[0.17, 0.10, 0.035]]).repeat(n, 1),
        contacts=torch.ones(n, 2, dtype=torch.bool),
        gripper_distance=torch.zeros(n),
        jaw_distance=torch.full((n,), 0.045),
        paper_yaw=torch.zeros(n),
        linear_velocity=torch.zeros(n, 3),
        angular_velocity=torch.zeros(n, 3),
    )
    p = SimpleNamespace(grasped=torch.ones(n, dtype=torch.bool), lifted=torch.zeros(n, dtype=torch.bool))
    env = SimpleNamespace(
        cfg=SimpleNamespace(box_size=(0.13, 0.045, 0.09), paper_size=(0.297, 0.21)),
        scene=SimpleNamespace(env_origins=torch.zeros(n, 3)),
        _box_initial_pose=torch.cat((position.clone(), q["quaternion"]), -1),
    )
    monkeypatch.setattr(mdp, "progress", lambda _: p)
    monkeypatch.setattr(mdp, "quantities", lambda _: (q, center, q["position"][:, 2] - 0.08))
    monkeypatch.setattr(mdp, "contact_force", lambda _, name: torch.full((n,), 0.981))
    return env, q, center, p


def test_approach_guidance_resumes_after_a_latched_grasp_loses_contact(monkeypatch):
    env, q, center, _ = fixture(monkeypatch)
    q["contacts"][:] = False
    center[:, 0] -= 0.10
    reward = mdp.ApproachReward.__new__(mdp.ApproachReward)
    reward.previous = torch.full((1,), float("nan"))
    assert reward(env).item() == 0
    center[:, 0] += 0.005
    assert reward(env).item() == pytest.approx(0.5, abs=1e-5)


def test_lift_guidance_is_nonzero_on_table_and_increases_for_upward_motion(monkeypatch):
    env, q, center, p = fixture(monkeypatch, n=3)
    q["position"][1, 2] += 0.01
    q["position"][2, 2] -= 0.005
    center[:] = q["position"]
    reward = mdp.transport_reward(env)
    assert reward[1] > reward[0] > reward[2]
    assert reward[0] > 0
    # Before clearance, moving sideways toward B must not improve the lift objective.
    q["position"][:, 2] = 0.08
    center[:] = q["position"]
    center[1, 1] += 0.05
    reward = mdp.transport_reward(env)
    assert reward[0] > reward[1]


def test_lifted_box_uses_b_target_and_receives_guidance_while_lowering(monkeypatch):
    env, q, center, p = fixture(monkeypatch, n=2)
    p.lifted[:] = True
    q["position"][1, 1] = 0.10
    center[:] = q["position"]
    reward = mdp.transport_reward(env)
    assert reward[1] > reward[0]
    assert reward[1] > 0  # removing the height multiplier also allows the final lowering stage


def test_lift_guidance_requires_current_bilateral_contact(monkeypatch):
    env, q, _, _ = fixture(monkeypatch)
    q["contacts"][:, 1] = False
    assert mdp.transport_reward(env).item() == 0


def test_contact_loss_and_reapproach_cycle_cannot_farm_approach_reward(monkeypatch):
    env, q, center, _ = fixture(monkeypatch)
    center[:, 0] -= 0.02
    reward = mdp.ApproachReward.__new__(mdp.ApproachReward)
    reward.previous = torch.full((1,), float("nan"))
    assert reward(env).item() == 0
    center[:, 0] -= 0.02
    total = reward(env).item()
    q["contacts"][:] = False
    for _ in range(4):
        center[:, 0] += 0.005
        total += reward(env).item()
    assert total <= 1e-4


def test_approach_reward_does_not_penalize_withdrawal_after_lift(monkeypatch):
    env, q, center, p = fixture(monkeypatch)
    p.lifted[:] = True
    q["contacts"][:] = False
    reward = mdp.ApproachReward.__new__(mdp.ApproachReward)
    reward.previous = torch.zeros(1)
    center[:, 0] += 0.10
    assert reward(env).item() == 0


def test_downward_pressure_penalty_subtracts_actual_box_weight(monkeypatch):
    masses = torch.tensor([[0.10], [0.20], [0.10]])
    env = SimpleNamespace(scene={"box": SimpleNamespace(data=SimpleNamespace(body_mass=SimpleNamespace(torch=masses)))})
    monkeypatch.setattr(mdp, "contact_force", lambda _, name: torch.tensor([0.981, 1.962, 10.981]))
    reward = mdp.table_pressure(env)
    assert reward.tolist() == pytest.approx([0, 0, 1])


def test_grasp_requires_two_consecutive_steps_and_partial_reset_clears_counter():
    from isaaclab_tutorial.tasks.place_box.progress import BoxProgress

    p = BoxProgress(2, "cpu")
    yes, no = torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
    p.update(yes, no, no)
    assert not p.grasped.any()
    p.update(torch.tensor([True, False]), no, no)
    assert p.grasped.tolist() == [True, False]
    p.reset(torch.tensor([0]))
    p.update(yes, no, no)
    assert not p.grasped.any()


def target_fixture(monkeypatch, n=1):
    env, q, center, p = fixture(monkeypatch, n)
    p.lifted[:] = True
    q["position"][:, :2] = q["paper_position"][:, :2]
    center[:] = q["position"]
    return env, q, center, p


@pytest.mark.parametrize("kind", ["TransportProgressReward", "ReleaseProgressReward"])
def test_stationary_target_hold_cannot_earn_repeated_progress_reward(monkeypatch, kind):
    env, q, _, _ = target_fixture(monkeypatch)
    q["jaw_distance"][:] = 0.0566
    reward = getattr(mdp, kind).__new__(getattr(mdp, kind))
    reward.previous = torch.full((1,), float("nan"))
    assert reward(env).item() == 0
    for _ in range(5):
        assert reward(env).item() <= 0


def test_open_release_and_withdraw_have_distinct_progress_credit(monkeypatch):
    env, q, _, _ = target_fixture(monkeypatch)
    reward = mdp.ReleaseProgressReward.__new__(mdp.ReleaseProgressReward)
    reward.previous = torch.full((1,), float("nan"))
    assert reward(env).item() == 0
    q["jaw_distance"][:] = 0.08
    assert reward(env).item() > 0
    q["contacts"][:] = False
    assert reward(env).item() > 0
    q["gripper_distance"][:] = 0.09
    assert reward(env).item() > 0
    assert reward(env).item() <= 0


def test_wide_jaws_still_touching_box_cannot_get_full_release_credit(monkeypatch):
    env, q, _, _ = target_fixture(monkeypatch)
    q["jaw_distance"][:] = 0.08
    q["gripper_distance"][:] = 0.09
    assert mdp.release_reward(env).item() == pytest.approx(0.25)
    q["contacts"][:] = False
    assert mdp.release_reward(env).item() == pytest.approx(1.0)


def test_release_regrasp_cycle_has_no_positive_net_progress_reward(monkeypatch):
    env, q, _, _ = target_fixture(monkeypatch)
    reward = mdp.ReleaseProgressReward.__new__(mdp.ReleaseProgressReward)
    reward.previous = torch.full((1,), float("nan"))
    reward(env)
    total = 0
    for _ in range(3):
        q["jaw_distance"][:] = 0.08
        q["contacts"][:] = False
        q["gripper_distance"][:] = 0.09
        total += reward(env).item()
        q["jaw_distance"][:] = 0.045
        q["contacts"][:] = True
        q["gripper_distance"][:] = 0.02
        total += reward(env).item()
    assert total <= 0


def test_release_potential_rejects_wrong_zone_hovering_and_excess_tilt(monkeypatch):
    import math

    env, q, _, _ = target_fixture(monkeypatch, 4)
    q["jaw_distance"][:] = 0.08
    q["contacts"][:] = False
    q["gripper_distance"][:] = 0.09
    q["position"][1, 0] += 0.20
    q["position"][2, 2] += 0.02
    q["quaternion"][3] = torch.tensor([math.sin(math.radians(20)), 0, 0, math.cos(math.radians(20))])
    assert mdp.release_reward(env).tolist() == pytest.approx([1, 0, 0, 0])


def test_potential_partial_reset_leaves_other_env_history_intact(monkeypatch):
    values = torch.tensor([0.2, 0.7])
    monkeypatch.setattr(mdp, "release_reward", lambda _: values)
    reward = mdp.ReleaseProgressReward.__new__(mdp.ReleaseProgressReward)
    reward.previous = torch.tensor([0.5, 0.6])
    reward.reset(torch.tensor([0]))
    assert reward(None).tolist() == pytest.approx([0, 0.995 * 0.7 - 0.6])


def test_moderately_tilted_box_can_open_but_needs_uprightness_for_final_retreat_credit(monkeypatch):
    import math

    from isaaclab_tutorial.tasks.place_box.geometry import box_corners

    env, q, _, _ = target_fixture(monkeypatch)
    q["jaw_distance"][:] = 0.08
    q["contacts"][:] = False
    q["gripper_distance"][:] = 0.09
    q["quaternion"][0] = torch.tensor([math.sin(math.radians(6.5)), 0, 0, math.cos(math.radians(6.5))])
    lowest = box_corners(q["position"], q["quaternion"])[..., 2].amin(1)
    q["position"][:, 2] += q["paper_position"][:, 2] - lowest
    assert mdp.release_reward(env).item() == pytest.approx(0.5)


def test_later_steps_do_not_overwrite_success_metrics_buffered_by_the_logger(monkeypatch):
    from isaaclab_tutorial.tasks.place_box.progress import BoxProgress

    env, q, _, _ = target_fixture(monkeypatch)
    p = BoxProgress(1, "cpu")
    p.grasped[:] = True
    p.lifted[:] = True
    p.stable_count[:] = 14
    q["contacts"][:] = False
    q["jaw_distance"][:] = 0.08
    q["gripper_distance"][:] = 0.09
    env.extras = {"log": {"Episode_Reward/success": 2.0}}
    monkeypatch.setattr(mdp, "progress", lambda _: p)
    term = mdp.BoxSuccess.__new__(mdp.BoxSuccess)
    assert term(env).item()
    # RSL-RL buffers this dictionary by reference rather than copying its entries.
    buffered = env.extras["log"]
    p.reset()
    assert not term(env).item()
    assert buffered["Metrics/success_rate"].item() == 1.0
    assert env.extras["log"]["Metrics/success_rate"].item() == 0.0
    assert env.extras["log"]["Episode_Reward/success"] == 2.0
