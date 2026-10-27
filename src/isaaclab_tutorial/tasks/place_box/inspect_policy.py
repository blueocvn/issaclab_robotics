"""Record deterministic box-policy playback, pad geometry and optional viewer frames."""

import argparse
import json
from pathlib import Path

import torch


def merge_terminal_state(current, terminal):
    """Replace auto-reset rows with their pre-reset terminal snapshot."""
    for env_id, values in terminal.items():
        for key, value in values.items():
            current[key][env_id] = value
    return current


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.utils.math import quat_apply_inverse

    from isaaclab_tasks.utils.hydra import resolve_presets

    from isaaclab_tutorial.tasks.place_vial.mdp.task import contact_force, fingertip_positions_w

    from .env import BoxEnv
    from .env_cfg import BoxEnvCfg
    from .mdp import progress, quantities

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="Exported TorchScript policy.pt")
    parser.add_argument("--output", type=Path, default=Path("logs/box_playback_trace.json"))
    parser.add_argument("--steps", type=int, default=899)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--frames", type=Path)
    parser.add_argument("--release-at", type=int, help="Diagnostic: override policy to open jaws then return home")
    add_launcher_args(parser)
    args = parser.parse_args()
    cfg = resolve_presets(BoxEnvCfg(), ["newton_mjwarp"])
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = 42
    policy = torch.jit.load(str(args.policy), map_location=cfg.sim.device).eval()
    rows = []
    with launch_simulation(cfg, args):
        env = BoxEnv(cfg)
        try:
            obs, _ = env.reset()

            def snapshot():
                q, center, lowest = quantities(env)
                fixed, moving = fingertip_positions_w(env)
                local = [quat_apply_inverse(q["quaternion"], pad - q["position"]) for pad in (fixed, moving)]
                p = progress(env)
                return dict(
                    box_position=q["position"].cpu().tolist(),
                    box_quaternion=q["quaternion"].cpu().tolist(),
                    bottom_height=lowest.cpu().tolist(),
                    pad_center=center.cpu().tolist(),
                    local_pads=torch.stack(local, 1).cpu().tolist(),
                    contacts=q["contacts"].cpu().tolist(),
                    fixed_force=contact_force(env, "fixed_jaw_contact").cpu().tolist(),
                    moving_force=contact_force(env, "moving_jaw_contact").cpu().tolist(),
                    table_force=contact_force(env, "box_table_contact").cpu().tolist(),
                    grasped=p.grasped.cpu().tolist(),
                    lifted=p.lifted.cpu().tolist(),
                    joint_position=env.scene["robot"].data.joint_pos.torch.cpu().tolist(),
                )

            terminal = {}
            episodes = [0] * env.num_envs
            original_reset = env._reset_idx
            release_start_joints = None

            def traced_reset(ids):
                values = snapshot()
                for env_id in ids.tolist():
                    terminal[env_id] = {key: value[env_id] for key, value in values.items()}
                original_reset(ids)

            # Diagnostic-only hook: inspect state before the regular automatic reset.
            env._reset_idx = traced_reset
            for step in range(args.steps):
                with torch.inference_mode():
                    actions = policy(obs["policy"]).clamp(-1, 1)
                    if args.release_at is not None and step >= args.release_at:
                        robot = env.scene["robot"]
                        measured = robot.data.joint_pos.torch
                        if release_start_joints is None:
                            release_start_joints = measured.clone()
                        # Open in place for two seconds, then slowly withdraw toward home.
                        alpha = min(1.0, max(0.0, (step - args.release_at - 60) / 180))
                        target = torch.lerp(release_start_joints, robot.data.default_joint_pos.torch, alpha)
                        gripper_id = robot.find_joints("gripper")[0][0]
                        target[:, gripper_id] = robot.data.soft_joint_pos_limits.torch[:, gripper_id, 1] - 0.05
                        scales = measured.new_full(measured.shape, 0.033)
                        scales[:, gripper_id] = 0.02
                        actions = ((target - measured) / scales).clamp(-1, 1)
                    obs, _, terminated, truncated, _ = env.step(actions)
                done = (terminated | truncated).cpu().tolist()
                if step % 15 == 0 or step == args.steps - 1 or any(done):
                    values = merge_terminal_state(snapshot(), terminal)
                    rows.append(
                        dict(
                            step=step + 1,
                            episode_ids=episodes.copy(),
                            **values,
                            actions=actions.cpu().tolist(),
                            done=done,
                            success=env.termination_manager.get_term("success").cpu().tolist(),
                        )
                    )
                for env_id, finished in enumerate(done):
                    episodes[env_id] += int(finished)
                terminal.clear()
                if args.frames and step in (0, 90, 300, 600, args.steps - 1):
                    from PIL import Image

                    env.sim.render()
                    visualizers = env.sim._visualizers
                    if not visualizers:
                        raise RuntimeError("--frames requires an RGB-capable visualizer")
                    pixels = visualizers[0].render_rgb_array()
                    if pixels is not None:
                        args.frames.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(pixels).save(args.frames / f"step_{step + 1:04d}.png")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(dict(policy=str(args.policy), scripted_release_at=args.release_at, trace=rows), indent=2)
            )
            print("BOX_PLAYBACK_TRACE=" + str(args.output), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
