"""Pure tensor geometry; all quaternions use Isaac Lab 3.0's XYZW convention."""

import math
from itertools import product

import torch

from isaaclab_tutorial.tasks.place_vial.mdp.geometry import quat_rotate_xyzw

BOX_SIZE = (0.13, 0.045, 0.09)
PAPER_SIZE = (0.297, 0.21)


def box_corners(position: torch.Tensor, quaternion: torch.Tensor, size=BOX_SIZE) -> torch.Tensor:
    """World positions of all eight cuboid corners, shape (N, 8, 3)."""
    offsets = position.new_tensor(list(product((-0.5, 0.5), repeat=3))) * position.new_tensor(size)
    q = quaternion[:, None, :].expand(-1, 8, -1)
    return position[:, None, :] + quat_rotate_xyzw(q, offsets[None].expand(len(position), -1, -1))


def placement_geometry(
    position,
    quaternion,
    paper_position,
    paper_yaw,
    size=BOX_SIZE,
    paper_size=PAPER_SIZE,
    margin=0.005,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-footprint containment, uprightness and bottom-to-table proximity.

    paper_position.z is the supporting tabletop, not the marker mesh's visual offset.
    Project every corner so a tilted upper corner cannot overhang unnoticed.
    """
    corners = box_corners(position, quaternion, size)
    delta = corners - paper_position[:, None, :]
    c, s = paper_yaw.cos()[:, None], paper_yaw.sin()[:, None]
    x = c * delta[..., 0] + s * delta[..., 1]
    y = -s * delta[..., 0] + c * delta[..., 1]
    inside = (x.abs() <= paper_size[0] / 2 - margin).all(1)
    inside &= (y.abs() <= paper_size[1] / 2 - margin).all(1)
    axis = quat_rotate_xyzw(quaternion, position.new_tensor((0, 0, 1)).expand_as(position))
    upright = axis[:, 2] >= math.cos(math.radians(10))
    supported = delta[..., 2].amin(1).abs() <= 0.003
    return inside, upright, supported


def placement_mask(
    position,
    quaternion,
    paper_position,
    paper_yaw,
    linear_velocity,
    angular_velocity,
    contacts,
    jaw_distance,
    gripper_distance,
    size=BOX_SIZE,
    paper_size=PAPER_SIZE,
    margin=0.005,
) -> torch.Tensor:
    """The unchanged full placement criterion: geometry, release, withdrawal and stillness."""
    inside, upright, supported = placement_geometry(
        position, quaternion, paper_position, paper_yaw, size, paper_size, margin
    )
    still = linear_velocity.norm(dim=-1) < 0.03
    still &= angular_velocity.norm(dim=-1) < 0.3
    released = ~contacts.any(-1) & (jaw_distance > size[1] + 0.005) & (gripper_distance > 0.08)
    return inside & upright & supported & still & released
