"""Box-specific reset, observation, reward and termination terms."""

import math

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_tutorial.tasks.place_vial.mdp.events import _ids, _yaw_quat
from isaaclab_tutorial.tasks.place_vial.mdp.task import contact_force, fingertip_positions_w

from .geometry import box_corners, placement_geometry, placement_mask
from .progress import BoxProgress


def progress(env):
    if not hasattr(env, "_box_progress"):
        env._box_progress = BoxProgress(env.num_envs, env.device, math.ceil(0.5 / env.step_dt))
    return env._box_progress


def quantities(env):
    box = env.scene["box"]
    pos, quat = box.data.root_pos_w.torch, box.data.root_quat_w.torch
    fixed, moving = fingertip_positions_w(env)
    center = (fixed + moving) / 2
    contacts = torch.stack(
        (contact_force(env, "fixed_jaw_contact") > 0.05, contact_force(env, "moving_jaw_contact") > 0.05), -1
    )
    paper = env.scene.env_origins + pos.new_tensor(env.cfg.scene.paper.init_state.pos)
    paper = paper.clone()
    paper[:, 2] = env.scene.env_origins[:, 2] + env.cfg.tabletop_z
    yaw = pos.new_full((env.num_envs,), env.cfg.paper_yaw)
    corners = box_corners(pos, quat, env.cfg.box_size)
    lowest = corners[..., 2].amin(1) - paper[:, 2]
    return (
        dict(
            position=pos,
            quaternion=quat,
            paper_position=paper,
            paper_yaw=yaw,
            linear_velocity=box.data.root_lin_vel_w.torch,
            angular_velocity=box.data.root_ang_vel_w.torch,
            contacts=contacts,
            jaw_distance=(fixed - moving).norm(dim=-1),
            gripper_distance=(center - pos).norm(dim=-1),
        ),
        center,
        lowest,
    )


def reset_home(
    env, env_ids, position_noise=0.01, yaw_noise=0.1, paper_yaw=0.0,
    nominal_fraction=0.0, reset_mode="random", sampling_seed=None, grid_start_index=0, replay_path="",
):
    """Independent home resets, using this box's geometry rather than vial dataset rows."""
    from .workspace import grid_offsets, replay_poses, sample_offsets, validate_distribution

    validate_distribution(position_noise, yaw_noise, nominal_fraction)
    if reset_mode not in ("random", "grid", "replay"):
        raise ValueError("reset_mode must be random, grid or replay")
    if reset_mode != "random" and nominal_fraction:
        raise ValueError("Grid/replay audits cannot mix nominal training starts")
    if not isinstance(grid_start_index, int) or grid_start_index < 0:
        raise ValueError("grid_start_index must be a nonnegative integer")
    ids = _ids(env, env_ids)
    generator = None
    if sampling_seed is not None:
        if not isinstance(sampling_seed, int) or sampling_seed < 0:
            raise ValueError("sampling_seed must be a nonnegative integer or None")
        if getattr(env, "_box_sampling_seed", None) != sampling_seed:
            env._box_sampling_generator = torch.Generator(device=env.device).manual_seed(sampling_seed)
            env._box_sampling_seed = sampling_seed
        generator = env._box_sampling_generator
    robot, box = env.scene["robot"], env.scene["box"]
    pose = box.data.default_root_pose.torch[ids].clone()
    case = torch.full(ids.shape, -1, device=env.device, dtype=torch.long)
    source = torch.ones_like(case)
    if reset_mode == "replay":
        if getattr(env, "_box_replay_path", None) != replay_path:
            env._box_replay_poses = replay_poses(replay_path, env.device)
            env._box_replay_path = replay_path
        case = ids.long() + grid_start_index
        pose = env._box_replay_poses[case % len(env._box_replay_poses)].clone()
        source[:] = 3
    else:
        if reset_mode == "grid":
            case = ids.long() + grid_start_index
            offsets = grid_offsets(case, position_noise, yaw_noise)
            source[:] = 2
        else:
            offsets, nominal = sample_offsets(
                len(ids), env.device, position_noise, yaw_noise, nominal_fraction, generator
            )
            source[nominal] = 0
        pose[:, :2] += offsets[:, :2]
        pose[:, 3:] = _yaw_quat(offsets[:, 2])
    joints = robot.data.default_joint_pos.torch[ids].clone()
    zeros = torch.zeros_like(joints)
    robot.write_joint_position_to_sim_index(position=joints, env_ids=ids)
    robot.write_joint_velocity_to_sim_index(velocity=zeros, env_ids=ids)
    robot.set_joint_position_target_index(target=joints, env_ids=ids)
    robot.set_joint_velocity_target_index(target=zeros, env_ids=ids)
    pose[:, :3] += env.scene.env_origins[ids]
    box.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    box.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
    progress(env).reset(ids)
    if not hasattr(env, "_box_initial_pose"):
        env._box_initial_pose = pose.new_zeros(env.num_envs, 7)
        env._box_reset_source = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        env._box_reset_case = torch.full_like(env._box_reset_source, -1)
    env._box_initial_pose[ids] = pose - torch.cat((env.scene.env_origins[ids], pose.new_zeros(len(ids), 4)), -1)
    env._box_reset_source[ids] = source
    env._box_reset_case[ids] = case
    env._box_reset_parameters = dict(
        position_noise=position_noise, yaw_noise=yaw_noise, nominal_fraction=nominal_fraction,
        reset_mode=reset_mode, sampling_seed=sampling_seed, grid_start_index=grid_start_index, replay_path=replay_path,
    )


def state_observation(env):
    """Privileged box state, target, pad positions and episode progress, in robot coordinates."""
    robot = env.scene["robot"]
    root_pos, root_quat = robot.data.root_link_pos_w.torch, robot.data.root_link_quat_w.torch
    q, center, lowest = quantities(env)
    fixed, moving = fingertip_positions_w(env)
    p = progress(env)
    # Local orientations are not needed: the world quaternion plus fixed base yaw fully specify orientation.
    return torch.cat(
        (
            robot.data.joint_pos.torch + getattr(env, "_so101_encoder_bias", 0.0),
            robot.data.joint_vel.torch,
            robot.data.joint_pos_target.torch,
            env.action_manager.action,
            quat_apply_inverse(root_quat, q["position"] - root_pos),
            q["quaternion"],
            quat_apply_inverse(root_quat, q["linear_velocity"]),
            quat_apply_inverse(root_quat, q["angular_velocity"]),
            quat_apply_inverse(root_quat, fixed - root_pos),
            quat_apply_inverse(root_quat, moving - root_pos),
            quat_apply_inverse(root_quat, q["paper_position"] - root_pos),
            q["paper_yaw"].sin()[:, None],
            q["paper_yaw"].cos()[:, None],
            q["contacts"].float(),
            torch.stack((p.grasped, p.lifted, p.placed), -1).float(),
            (p.stable_count.float() / p.stable_steps)[:, None],
        ),
        -1,
    )


class BoxSuccess(ManagerTermBase):
    def reset(self, env_ids=None):
        progress(self._env).reset(env_ids)

    def __call__(self, env):
        q, _, lowest = quantities(env)
        p = progress(env)
        holding = q["contacts"].all(-1) & (q["gripper_distance"] < 0.10)
        placed = placement_mask(**q, size=env.cfg.box_size, paper_size=env.cfg.paper_size)
        placed &= contact_force(env, "box_table_contact") > 0.05
        success = p.update(holding, holding & (lowest >= 0.03), placed)
        # Preserve terminal rows before automatic partial reset overwrites live progress.
        env._box_terminal_progress = torch.stack((p.grasped, p.lifted, p.placed, p.success), -1).clone()
        inside, upright, near_table = placement_geometry(
            q['position'], q['quaternion'], q['paper_position'], q['paper_yaw'],
            size=env.cfg.box_size, paper_size=env.cfg.paper_size,
        )
        on_target = inside & near_table & (contact_force(env, 'box_table_contact') > 0.05)
        released = on_target & ~q['contacts'].any(-1) & (q['jaw_distance'] > env.cfg.box_size[1] + 0.005)
        env._box_terminal_details = {
            'on_target': on_target.clone(),
            'released': released.clone(),
            'withdrawn': (released & (q['gripper_distance'] > 0.08)).clone(),
            'upright': (on_target & upright).clone(),
        }
        # RSL-RL buffers this dictionary by reference; publish a new snapshot each step.
        log = env.extras["log"] = dict(env.extras.get("log", {}))
        for name, value in zip(
            ("grasp", "lift", "placement", "success"), env._box_terminal_progress.unbind(-1), strict=True
        ):
            log[f"Metrics/{name}_rate"] = value.float().mean()
        for name, value in env._box_terminal_details.items():
            log[f'Metrics/{name}_rate'] = value.float().mean()
        log["Metrics/current_contact_rate"] = holding.float().mean()
        log["Metrics/box_bottom_height_m"] = lowest.mean()
        log["Metrics/box_table_force_n"] = contact_force(env, "box_table_contact").mean()
        return success


def box_dropped(env):
    """Fail when the box falls below the supporting tabletop."""
    q, _, _ = quantities(env)
    return q["position"][:, 2] < q["paper_position"][:, 2] - 0.02


def box_out_of_bounds(env):
    pos = env.scene["box"].data.root_pos_w.torch - env.scene.env_origins
    return (pos[:, :2].abs() > 0.65).any(-1) | (pos[:, 2] > 0.8) | ~torch.isfinite(pos).all(-1)


class ApproachReward(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous = torch.full((env.num_envs,), float("nan"), device=env.device)

    def reset(self, env_ids=None):
        self.previous[slice(None) if env_ids is None else env_ids] = float("nan")

    def __call__(self, env):
        q, center, _ = quantities(env)
        distance = (center - q["position"]).norm(dim=-1)
        # Unclipped potential differences keep a retreat/reapproach cycle from paying.
        reward = ((self.previous - distance) / 0.01).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        self.previous.copy_(distance)
        # A historical grasp flag must not disable recovery after losing the box.
        holding = q["contacts"].all(-1) & (q["gripper_distance"] < 0.10)
        # Preserve retreat costs during contact; suppressing them would let a
        # stationary-box jaw-slide/reapproach cycle farm the positive half.
        reward = torch.where(holding, reward.clamp(max=0.0), reward)
        # After physical lift, carry/release shaping takes over; withdrawal
        # from a successfully placed box must not incur an approach penalty.
        return torch.where(progress(env).lifted, 0.0, reward)


def transport_reward(env):
    """Within-episode curriculum: lift above A first, then carry/lower at B.

    The former lowest-point multiplier made guidance zero on the table, and
    aimed at B before lifting. Both left a contact-only local optimum.
    Reward box and pad motion together, so sliding the jaws up a stationary
    box cannot achieve the full lift reward. Full success still requires a
    physical 3 cm lift, upright placement, release and stable table support.
    """
    q, center, _ = quantities(env)
    p = progress(env)
    # Keep XY over the current box, making lift guidance independent of hidden reset history.
    lift_goal = q["position"].clone()
    lift_goal[:, 2] = q["paper_position"][:, 2] + env.cfg.box_size[2] / 2 + 0.10
    place_goal = q["paper_position"].clone()
    place_goal[:, 2] += env.cfg.box_size[2] / 2
    goal = torch.where(p.lifted[:, None], place_goal, lift_goal)
    error = 0.5 * ((q["position"] - goal).norm(dim=-1) + (center - goal).norm(dim=-1))
    # XYZW quaternion has local +Z alignment 1 - 2*(x*x + y*y).
    alignment = (1 - 2 * q["quaternion"][:, :2].square().sum(-1)).clamp(0, 1)
    holding = q["contacts"].all(-1) & (q["gripper_distance"] < 0.10)
    return torch.where(p.grasped & holding, torch.exp(-error / 0.08) * alignment, 0.0)


def table_pressure(env, free_force: float = 2.0, force_span: float = 8.0):
    """Penalize pushing down, excluding actual box weight and a small contact allowance."""
    weight = env.scene["box"].data.body_mass.torch.sum(-1) * 9.81
    force = contact_force(env, "box_table_contact")
    return ((force - weight - free_force) / force_span).clamp(0, 1)


def release_reward(env):
    """Bounded opening/release/withdrawal potential, not a repeating step bonus."""
    q, _, _ = quantities(env)
    inside, upright, supported = placement_geometry(
        q["position"],
        q["quaternion"],
        q["paper_position"],
        q["paper_yaw"],
        size=env.cfg.box_size,
        paper_size=env.cfg.paper_size,
    )
    supported &= contact_force(env, "box_table_contact") > 0.05
    alignment = 1 - 2 * q["quaternion"][:, :2].square().sum(-1)
    # Permit opening at the observed 13-degree held tilt so gravity can seat the box.
    # The final withdrawal credit and success still require the original 10-degree limit.
    ready = progress(env).lifted & inside & supported & (alignment >= math.cos(math.radians(25)))
    opening = ((q["jaw_distance"] - env.cfg.box_size[1]) / 0.035).clamp(0, 1)
    released = ~q["contacts"].any(-1) & (q["jaw_distance"] > env.cfg.box_size[1] + 0.005)
    retreat = ((q["gripper_distance"] - 0.03) / 0.05).clamp(0, 1)
    potential = 0.25 * opening + 0.25 * released.float()
    potential += 0.5 * (released & upright).float() * retreat
    return torch.where(ready, potential, 0.0)


class PotentialProgressReward(ManagerTermBase):
    """Signed potential differences: stationary poses and return cycles cannot pay.

    Use the PPO discount 0.995; retain negative differences when contact, support
    or target containment is lost. The first sample after each partial reset is
    zero. No reward is derived from commanded opening without physical motion.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous = torch.full((env.num_envs,), float("nan"), device=env.device)

    def reset(self, env_ids=None):
        self.previous[slice(None) if env_ids is None else env_ids] = float("nan")

    def difference(self, potential):
        reward = (0.995 * potential - self.previous).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        self.previous.copy_(potential)
        return reward


class TransportProgressReward(PotentialProgressReward):
    def __call__(self, env):
        return self.difference(transport_reward(env))


class ReleaseProgressReward(PotentialProgressReward):
    def __call__(self, env):
        return self.difference(release_reward(env))


class MilestoneReward(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.paid = torch.zeros(env.num_envs, 3, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self.paid[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env):
        p = progress(env)
        reached = torch.stack((p.grasped, p.lifted, p.placed), -1)
        result = ((reached & ~self.paid).float() * reached.new_tensor((1, 2, 4), dtype=torch.float)).sum(-1)
        self.paid.copy_(reached)
        return result


def success_reward(env):
    return progress(env).success.float()
