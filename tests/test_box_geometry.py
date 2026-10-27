"""Placement must use the complete cuboid, including rotation and release."""

import math

import torch

from isaaclab_tutorial.tasks.place_box.geometry import box_corners, placement_mask
from isaaclab_tutorial.tasks.place_box.progress import BoxProgress


def state(n=1):
    return dict(
        position=torch.tensor([[0.0, 0.0, 0.045]]).repeat(n, 1),
        quaternion=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1),
        paper_position=torch.zeros(n, 3),
        paper_yaw=torch.zeros(n),
        linear_velocity=torch.zeros(n, 3),
        angular_velocity=torch.zeros(n, 3),
        contacts=torch.zeros(n, 2, dtype=torch.bool),
        jaw_distance=torch.full((n,), 0.1),
        gripper_distance=torch.full((n,), 0.15),
    )


def test_box_dimensions_match_photo_orientation():
    s = state()
    corners = box_corners(s["position"], s["quaternion"])
    torch.testing.assert_close(corners.amax(1) - corners.amin(1), torch.tensor([[0.13, 0.045, 0.09]]))


def test_whole_footprint_not_just_center_must_fit():
    s = state(3)
    s["position"][:, 0] = torch.tensor([0.0, 0.078, 0.085])
    assert placement_mask(**s).tolist() == [True, True, False]


def test_rotated_paper_and_box_are_compared_in_paper_frame():
    s = state(2)
    s["paper_yaw"][:] = math.pi / 2
    s["quaternion"][:] = torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])
    s["position"][:, 1] = torch.tensor([0.078, 0.085])
    assert placement_mask(**s).tolist() == [True, False]


def test_held_hovering_tipped_moving_or_unwithdrawn_box_cannot_succeed():
    s = state(7)
    s["contacts"][1, 0] = True
    s["position"][2, 2] += 0.03
    s["quaternion"][3] = torch.tensor([math.sin(math.radians(15)), 0, 0, math.cos(math.radians(15))])
    s["linear_velocity"][4, 0] = 0.04
    s["gripper_distance"][5] = 0.01
    s["jaw_distance"][6] = 0.04
    assert placement_mask(**s).tolist() == [True, False, False, False, False, False, False]


def test_success_requires_grasp_lift_and_fifteen_consecutive_stable_steps():
    p = BoxProgress(2, "cpu", stable_steps=15)
    yes, no = torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
    p.update(no, no, yes)
    assert not p.success.any()  # spawning on paper is not a completed pick-and-place
    p.update(yes, no, no)
    p.update(yes, yes, no)
    for _ in range(14):
        p.update(no, no, yes)
    assert not p.success.any()
    p.update(no, no, torch.tensor([False, True]))
    assert p.success.tolist() == [False, True]
    p.reset(torch.tensor([1]))
    assert p.grasped.tolist() == [True, False]
    assert p.stable_count.tolist() == [0, 0]


def test_all_eight_corners_are_checked_for_tilted_box_overhang():
    s = state()
    s["quaternion"][0] = torch.tensor([0, math.sin(math.radians(4)), 0, math.cos(math.radians(4))])
    corners = box_corners(s["position"], s["quaternion"])
    s["position"][0, 2] -= corners[..., 2].amin()
    s["position"][0, 0] = 0.078
    assert not placement_mask(**s).item()
