"""Initialize and run bounded Camera PPO fine-tuning from a distilled student."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch


def initialize_from_student(runner, checkpoint):
    """Copy complete student state into a fresh actor; leave PPO critic/Adam alone."""
    if runner.current_learning_iteration != 0 or runner.alg.optimizer.state:
        raise ValueError("Camera PPO initialization requires a fresh runner and optimizer")
    source = checkpoint.get("student_state_dict")
    target = runner.alg.actor.state_dict()
    if not isinstance(source, dict) or source.keys() != target.keys():
        raise ValueError("Student and PPO actor state keys differ")
    for key, value in source.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != target[key].shape
            or value.dtype != target[key].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid student tensor: {key}")
    runner.alg.actor.load_state_dict(source, strict=True)


def create_camera_ppo_runner(env, agent_cfg, log_dir=None):
    """Apply Isaac Lab's installed-version migration before native construction."""
    from isaaclab_rl.rsl_rl.utils import (
        check_rsl_rl_version,
        create_rsl_rl_runner,
        handle_deprecated_rsl_rl_cfg,
    )

    handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    return create_rsl_rl_runner(env, agent_cfg, log_dir=log_dir)


def main(argv=None):
    from rsl_rl.models import CNNModel

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.utils.hydra import resolve_presets

    from .agents import BoxCameraPPORunnerCfg
    from .env import BoxEnv
    from .env_cfg import BoxCameraEnvCfg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=128)
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    if not 1 <= args.iterations <= 50 or args.num_envs <= 0:
        parser.error("Use1-50 iterations and a positive environment count")
    if args.output.exists():
        parser.error("Use a new output directory")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    torch.set_num_threads(4)
    torch.manual_seed(67)
    torch.cuda.manual_seed_all(67)
    cfg = resolve_presets(BoxCameraEnvCfg(), ["newton_mjwarp", "newton_renderer"])
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 67
    cfg.sim.device = args.device or "cuda:0"
    cfg.configure_layout()
    cfg.events.reset_home.params.update(
        position_noise=0.1, yaw_noise=3.141592653589793, nominal_fraction=0.2, sampling_seed=67
    )
    agent = BoxCameraPPORunnerCfg()
    agent.device = cfg.sim.device
    args.output.mkdir(parents=True)
    (args.output / "settings.json").write_text(
        json.dumps(
            dict(
                agent=agent.to_dict(),
                resets=cfg.events.reset_home.params,
                source_checkpoint=str(args.checkpoint),
                source_sha256=source_hash,
                iterations=args.iterations,
                num_envs=args.num_envs,
            ),
            indent=2,
            default=str,
        )
        + "\n"
    )
    with launch_simulation(cfg, args):
        env = RslRlVecEnvWrapper(BoxEnv(cfg), clip_actions=1.0)
        try:
            runner = create_camera_ppo_runner(env, agent, log_dir=str(args.output / "training"))
            critic_before = {k: v.clone() for k, v in runner.alg.critic.state_dict().items()}
            initialize_from_student(runner, checkpoint)
            obs = env.get_observations()
            reference_cfg = copy.deepcopy(to_dict(agent.actor))
            reference_cfg.pop("class_name")
            reference = (
                CNNModel(obs, {"student": ["wrist_rgb", "proprioception"]}, "student", 6, **reference_cfg)
                .to(env.device)
                .eval()
            )
            reference.load_state_dict(checkpoint["student_state_dict"])
            actor = runner.get_inference_policy()
            with torch.inference_mode():
                maximum_error = float((actor(obs) - reference(obs)).abs().max())
            assert maximum_error == 0.0, "Pre-update actor actions changed"
            assert all(torch.equal(v, critic_before[k]) for k, v in runner.alg.critic.state_dict().items())
            assert not runner.alg.optimizer.state
            assert all(torch.equal(v.cpu(), checkpoint["student_state_dict"][k]) for k, v in actor.state_dict().items())
            initial = args.output / "initial.pt"
            runner.save(str(initial), infos={"source_sha256": source_hash, "pre_update_action_error": maximum_error})
            initial_count = int(actor.state_dict()["obs_normalizer.count"])
            (args.output / "initialization.json").write_text(
                json.dumps(
                    dict(
                        source_sha256=source_hash,
                        action_max_error=maximum_error,
                        actor_all_tensors_equal=True,
                        critic_unchanged=True,
                        optimizer_fresh=True,
                        normalizer_count=initial_count,
                    ),
                    indent=2,
                )
                + "\n"
            )
            del reference
            runner.learn(args.iterations, init_at_random_ep_len=False)
            endpoint = args.output / "training" / f"model_{args.iterations - 1}.pt"
            saved = torch.load(endpoint, map_location="cpu", weights_only=False)
            assert all(
                torch.isfinite(v).all()
                for name in ("actor_state_dict", "critic_state_dict")
                for v in saved[name].values()
            )
            counts = sorted({int(v["step"]) for v in saved["optimizer_state_dict"]["state"].values() if "step" in v})
            expected = args.iterations * agent.algorithm.num_learning_epochs * agent.algorithm.num_mini_batches
            assert counts == [expected]
            assert (
                int(saved["actor_state_dict"]["obs_normalizer.count"])
                == initial_count + args.iterations * args.num_envs * agent.num_steps_per_env
            )
            assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == source_hash
            verification = dict(
                source_sha256=source_hash,
                checkpoint=str(endpoint),
                checkpoint_sha256=hashlib.sha256(endpoint.read_bytes()).hexdigest(),
                iterations=args.iterations,
                optimizer_steps=counts,
                finite_actor_and_critic=True,
                actor_normalizer_count=int(saved["actor_state_dict"]["obs_normalizer.count"]),
                initial_normalizer_count=initial_count,
                algorithm="PPO",
                actor_inputs=["wrist_rgb", "proprioception"],
                critic_inputs=["critic"],
            )
            (args.output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
            print(json.dumps(verification, indent=2), flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    main()
