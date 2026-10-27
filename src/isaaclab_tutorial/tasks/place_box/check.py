"""Smoke, camera and physical grasp/reach probes for the box task.

Run: uv run python -m isaaclab_tutorial.tasks.place_box.check --device cuda:0 --visualizer none
Add --grasp-probe for independent FK/IK plus a gravity/contact lift test, or --camera for RGB checks.
This diagnostic never writes a policy checkpoint or fabricates object motion during the lift.
"""

import argparse
import json

import numpy as np
import torch


def grasp_probe(env):
    """Find pad poses with the actual robot model, then test a contact-driven grasp and lift."""
    import newton
    import warp as wp
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    from isaaclab import cloner
    from isaaclab_newton.cloner import copy_newton_clone_source

    from isaaclab_tutorial.tasks.place_vial.mdp.task import FIXED_FINGERTIP_OFFSET, MOVING_FINGERTIP_OFFSET

    from .mdp import quantities

    robot = env.scene["robot"]
    plan = env.sim.get_clone_plan()
    prototypes = cloner.path.get_asset_prototype_paths(plan)
    ids = cloner.path.get_asset_prototypes(plan, robot.cfg.prim_path)
    paths = {prototypes[index] for index in ids if prototypes[index] is not None}
    if len(paths) != 1:
        raise RuntimeError("Expected one SO-101 prototype")
    builder = copy_newton_clone_source(paths.pop())
    model = builder.finalize(device="cpu")
    state = model.state()
    q_wp = wp.clone(model.joint_q)
    q_np = wp.to_torch(q_wp).numpy()
    joint_names = [str(label).rsplit("/", 1)[-1] for label in model.joint_label]
    starts = wp.to_torch(model.joint_q_start).numpy()
    coordinates = [int(starts[joint_names.index(name)]) for name in robot.joint_names]
    bodies = [str(label).rsplit("/", 1)[-1] for label in model.body_label]
    body_ids = [bodies.index("gripper"), bodies.index("moving_jaw_so101_v1")]
    limits = robot.data.soft_joint_pos_limits.torch[0].cpu().numpy()
    lower, upper = limits[:, 0] + 0.005, limits[:, 1] - 0.005
    rng = np.random.default_rng(42)

    def pads(joints):
        q_np[coordinates] = joints
        newton.eval_fk(model, q_wp, model.joint_qd, state)
        poses = wp.to_torch(state.body_q).numpy()
        return np.array(
            [
                poses[i, :3] + Rotation.from_quat(poses[i, 3:]).apply(offset)
                for i, offset in zip(body_ids, (FIXED_FINGERTIP_OFFSET, MOVING_FINGERTIP_OFFSET), strict=True)
            ]
        )

    def solve(center, span, seed, attempts=8):
        desired = np.array([center + [0, span / 2, 0], center - [0, span / 2, 0]])
        best = None
        for n in range(attempts):
            initial = np.clip(seed if n == 0 else rng.uniform(lower, upper), lower, upper)
            result = least_squares(
                lambda x: (pads(x) - desired).reshape(-1),
                initial,
                bounds=(lower, upper),
                diff_step=1e-3,
                max_nfev=150,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )
            error = float(np.linalg.norm(pads(result.x) - desired, axis=1).max())
            if best is None or error < best[0]:
                best = error, result.x.copy()
            if error < 0.002:
                break
        if best[0] > 0.005:
            raise RuntimeError(f"Pad IK unreachable: target={center.tolist()}, error={best[0]:.4f} m")
        return best[1], best[0]

    origin = env.scene.env_origins[0].cpu().numpy()
    center = env.scene["box"].data.root_pos_w.torch[0].cpu().numpy() - origin
    seed = robot.data.default_joint_pos.torch[0].cpu().numpy()
    open_q, error_a = solve(center, 0.065, seed)
    closed_q, _ = solve(center, 0.043, open_q)
    lifted_center = center + np.array([0, 0, 0.10])
    lift_q, error_lift = solve(lifted_center, 0.043, closed_q)
    b = np.array(env.cfg.scene.paper.init_state.pos)
    b[2] = lifted_center[2]
    _, error_b = solve(b, 0.043, lift_q)

    def command(joints):
        return torch.as_tensor(joints, dtype=torch.float32, device=env.device)[None]

    def simulate(start, end, steps):
        for alpha in np.linspace(0, 1, steps):
            robot.set_joint_position_target_index(target=command(start * (1 - alpha) + end * alpha))
            env.scene.write_data_to_sim()
            env.sim.step()
            env.scene.update(env.physics_dt)

    # Only the robot is positioned for this isolated contact probe. The box remains dynamic throughout.
    robot.write_joint_position_to_sim_index(position=command(open_q))
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(1, 6, device=env.device))
    robot.set_joint_position_target_index(target=command(open_q))
    env.sim.forward()
    simulate(open_q, open_q, 30)
    simulate(open_q, closed_q, 180)
    before, _, _ = quantities(env)
    bilateral_before = bool(before["contacts"].all())
    simulate(closed_q, lift_q, 300)
    simulate(lift_q, lift_q, 60)
    after, _, lowest = quantities(env)
    result = dict(
        ik_error_a_m=error_a,
        ik_error_lift_m=error_lift,
        ik_error_b_m=error_b,
        bilateral_grasp=bilateral_before,
        bilateral_after_lift=bool(after["contacts"].all()),
        box_bottom_height_m=float(lowest[0]),
        box_position_m=after["position"][0].cpu().tolist(),
        box_quaternion_xyzw=after["quaternion"][0].cpu().tolist(),
        joint_tracking_error_rad=(robot.data.joint_pos.torch[0] - command(lift_q)[0]).cpu().tolist(),
        lifted=bool((lowest >= 0.03).all()),
    )
    result["passed"] = result["bilateral_grasp"] and result["bilateral_after_lift"] and result["lifted"]
    return result


def main():
    from isaaclab.app import add_launcher_args, launch_simulation

    from isaaclab_tasks.utils.hydra import resolve_presets

    from .env import BoxEnv
    from .env_cfg import BoxCameraEnvCfg, BoxEnvCfg
    from .mdp import quantities

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--grasp-probe", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sim2real", action="store_true")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.camera and args.grasp_probe:
        parser.error("--camera and --grasp-probe are separate checks")
    presets = ["newton_mjwarp"] + (["newton_renderer"] if args.camera else [])
    if args.sim2real:
        presets += ["sim2real"]
    cfg = resolve_presets(BoxCameraEnvCfg() if args.camera else BoxEnvCfg(), presets)
    cfg.scene.num_envs = 1 if args.grasp_probe else 8
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = 42
    cfg.configure_layout()
    cfg.events.reset_home.params.update(position_noise=0.0, yaw_noise=0.0)
    with launch_simulation(cfg, args):
        env = BoxEnv(cfg)
        try:
            obs, _ = env.reset()
            actions = torch.zeros(env.num_envs, 6, device=env.device)
            for _ in range(args.steps):
                obs, _, _, _, _ = env.step(actions)
            q, _, lowest = quantities(env)
            result = dict(
                device=env.device,
                num_envs=env.num_envs,
                box_size_m=cfg.box_size,
                paper_size_m=cfg.paper_size,
                box_bottom_height_m=lowest.cpu().tolist(),
                finite_observations=all(bool(torch.isfinite(v).all()) for v in obs.values()),
            )
            if not result["finite_observations"] or not bool((lowest.abs() < 0.005).all()):
                raise RuntimeError(f"Invalid resting scene: {result}")
            if args.camera:
                image = obs["wrist_rgb"]
                result.update(rgb_shape=list(image.shape), rgb_range=[float(image.min()), float(image.max())])
                if float(image.max() - image.min()) < 0.05:
                    raise RuntimeError("Camera image has no visible contrast")
            if args.grasp_probe:
                result["grasp_probe"] = grasp_probe(env)
            print("SO101_BOX_CHECK=" + json.dumps(result, sort_keys=True), flush=True)
            if args.grasp_probe and not result["grasp_probe"]["passed"]:
                return 1
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
