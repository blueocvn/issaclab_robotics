"""Sampling for optional wider training resets and controlled workspace audits."""

import json
import math
from pathlib import Path

import torch

STAGES = {
    "A": (0.02, 0.1),
    "B": (0.05, 0.1),
    "C": (0.1, 0.1),
    "D": (0.1, math.pi / 4),
    "E": (0.1, math.pi / 2),
    "F": (0.1, math.pi),
}
SOURCES = {0: "nominal", 1: "current", 2: "grid", 3: "replay"}
GRID_CASES = 200


def validate_distribution(position_noise, yaw_noise, nominal_fraction=0):
    if not all(math.isfinite(x) for x in (position_noise, yaw_noise, nominal_fraction)):
        raise ValueError("Reset distribution parameters must be finite")
    if position_noise < 0 or not 0 <= yaw_noise <= math.pi or not 0 <= nominal_fraction <= 1:
        raise ValueError("Reset position/yaw ranges and nominal fraction are invalid")


def sample_offsets(count, device, position_noise, yaw_noise, nominal_fraction=0, generator=None):
    """Return XY/yaw offsets and nominal membership; default preserves the old RNG draws."""
    validate_distribution(position_noise, yaw_noise, nominal_fraction)
    if not isinstance(count, int) or count < 0:
        raise ValueError("Reset count must be a nonnegative integer")
    xy = torch.empty(count, 2, device=device).uniform_(-position_noise, position_noise, generator=generator)
    yaw = torch.empty(count, device=device).uniform_(-yaw_noise, yaw_noise, generator=generator)
    nominal = torch.zeros(count, device=device, dtype=torch.bool)
    if nominal_fraction:
        nominal = torch.rand(count, device=device, generator=generator) < nominal_fraction
        # Draw full tensors so no host synchronization is needed to count selected CUDA environments.
        nominal_xy = torch.empty_like(xy).uniform_(-0.01, 0.01, generator=generator)
        nominal_yaw = torch.empty_like(yaw).uniform_(-0.1, 0.1, generator=generator)
        xy = torch.where(nominal[:, None], nominal_xy, xy)
        yaw = torch.where(nominal, nominal_yaw, yaw)
    return torch.cat((xy, yaw[:, None]), -1), nominal


def grid_offsets(indices, position_noise, yaw_noise):
    """5x5 XY centers (including edges), eight yaw angles,200 cases per repeat.

    Full yaw is [-pi,pi), avoiding duplicate -pi/+pi orientations. Smaller yaw
    ranges use the same evenly spaced half-open convention.
    """
    validate_distribution(position_noise, yaw_noise)
    if (indices < 0).any():
        raise ValueError("Grid indices must be nonnegative")
    case = indices % GRID_CASES
    return torch.stack(
        (
            ((case // 40).float() / 2 - 1) * position_noise,
            (((case // 8) % 5).float() / 2 - 1) * position_noise,
            ((case % 8).float() / 4 - 1) * yaw_noise,
        ),
        -1,
    )


def replay_poses(path, device):
    """Read local-frame XYZW poses; reject nonfinite or nonunit quaternions."""
    data = json.loads(Path(path).read_text())
    poses = torch.tensor(data["poses"], dtype=torch.float32, device=device)
    if poses.ndim != 2 or poses.shape[1] != 7 or not len(poses) or not torch.isfinite(poses).all():
        raise ValueError("Replay poses must be a nonempty finite Nx7 array")
    if not torch.allclose(poses[:, 3:].norm(dim=-1), torch.ones(len(poses), device=device), atol=1e-5):
        raise ValueError("Replay pose quaternions must be normalized")
    return poses
