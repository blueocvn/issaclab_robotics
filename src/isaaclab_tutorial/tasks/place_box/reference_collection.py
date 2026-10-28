"""Collect successful frozen-teacher first episodes and freeze a reference bank."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .distillation import STUDENT_KEYS, phase_ids
from .reference_replay import ReferenceBank, build_reference, teacher_fingerprint


class FirstEpisodeReferences:
    """Sparse pre-action snapshots; ignore automatically reset later episodes."""

    def __init__(self, num_envs, population, stride=4):
        if stride <= 0 or population not in ("nominal", "wide"):
            raise ValueError("Invalid reference collection settings")
        self.population, self.stride = population, stride
        self.active = [True] * num_envs
        self.rows = [[] for _ in range(num_envs)]
        self.episodes = []

    @property
    def done(self):
        return not any(self.active)

    def capture(self, obs, labels, phases, step):
        if step % self.stride:
            return
        values = {
            key: obs[key].detach().to("cpu", dtype=torch.float16 if key == "wrist_rgb" else torch.float32)
            for key in STUDENT_KEYS
        }
        values.update(labels=labels.detach().cpu(), phases=phases.detach().cpu())
        for env_id, active in enumerate(self.active):
            if active:
                self.rows[env_id].append({key: value[env_id].clone() for key, value in values.items()})

    def finish(self, dones, successes):
        for env_id, done in enumerate(dones.bool().flatten().cpu().tolist()):
            if done and self.active[env_id]:
                rows = self.rows[env_id]
                values = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
                self.episodes.append(
                    dict(
                        population=self.population,
                        success=bool(successes[env_id]),
                        observations={key: values[key] for key in STUDENT_KEYS},
                        labels=values["labels"],
                        phases=values["phases"],
                        env_id=env_id,
                    )
                )
                self.active[env_id] = False
                self.rows[env_id] = []


def merge_collections(nominal, wide, capacity_per_phase=1024, seed=43):
    """Require two matching teacher identities and keep only successful episodes."""
    if nominal["population"] != "nominal" or wide["population"] != "wide":
        raise ValueError("Reference collection population mismatch")
    if nominal["teacher_sha256"] != wide["teacher_sha256"]:
        raise ValueError("Reference collections use different teachers")
    for data in (nominal, wide):
        if data["teacher_sha256"] != teacher_fingerprint(data["teacher_state_dict"]):
            raise ValueError("Reference teacher provenance mismatch")
    bank = build_reference(
        nominal["episodes"] + wide["episodes"],
        nominal["teacher_state_dict"],
        capacity_per_phase=capacity_per_phase,
        seed=seed,
    )
    bank["collection_provenance"] = [data["provenance"] for data in (nominal, wide)]
    ReferenceBank(bank)  # Validate before saving a consumable bank.
    return bank


def main(argv=None):
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--population", choices=("nominal", "wide"))
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--nominal", type=Path)
    parser.add_argument("--wide", type=Path)
    parser.add_argument("--capacity-per-phase", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output must be a new file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.nominal or args.wide:
        if not (args.nominal and args.wide) or args.population or args.checkpoint:
            parser.error("Merge requires --nominal and --wide only")
        nominal, wide = [torch.load(p, map_location="cpu", weights_only=False) for p in (args.nominal, args.wide)]
        bank = merge_collections(nominal, wide, args.capacity_per_phase, args.seed)
        torch.save(bank, args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "successful_episodes": bank["successful_episodes"],
                    "sizes": {p: v["sizes"] for p, v in bank["populations"].items()},
                },
                indent=2,
            )
        )
        return 0
    if not args.checkpoint or not args.population or args.num_envs <= 0 or args.episodes <= 0:
        parser.error("Collection requires checkpoint, population and positive episode/env counts")
    if args.episodes % args.num_envs:
        parser.error("Episodes must be divisible by num-envs")

    from rsl_rl.models import MLPModel

    from isaaclab_tasks.utils.hydra import resolve_presets

    from .env import BoxEnv
    from .env_cfg import BoxCameraEnvCfg
    from .mdp import progress

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    teacher_state = checkpoint["teacher_state_dict"]
    cfg = resolve_presets(BoxCameraEnvCfg(), ["newton_mjwarp", "newton_renderer"])
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = args.seed
    cfg.configure_layout()
    cfg.events.reset_home.params.update(
        position_noise=0.01 if args.population == "nominal" else 0.1,
        yaw_noise=0.1 if args.population == "nominal" else 3.141592653589793,
        nominal_fraction=0.0,
        sampling_seed=args.seed,
    )
    episodes, starts = [], []
    with launch_simulation(cfg, args):
        env = BoxEnv(cfg)
        try:
            teacher = None
            for batch in range(args.episodes // args.num_envs):
                obs, _ = env.reset()
                starts.append(env.scene["box"].data.root_state_w.torch[:, :7].cpu().tolist())
                if teacher is None:
                    teacher = MLPModel(
                        obs,
                        {"teacher": ["teacher_state"]},
                        "teacher",
                        6,
                        hidden_dims=[256, 256, 128],
                        activation="elu",
                        obs_normalization=True,
                        distribution_cfg={
                            "class_name": "GaussianDistribution",
                            "init_std": 0.2,
                            "std_type": "log",
                            "std_range": (0.05, 0.3),
                        },
                    ).to(env.device)
                    teacher.load_state_dict(teacher_state)
                    teacher.eval()
                collector = FirstEpisodeReferences(env.num_envs, args.population, args.stride)
                for step in range(int(env.max_episode_length) + 2):
                    with torch.inference_mode():
                        labels = teacher(obs).clamp(-1, 1)
                        collector.capture(obs, labels, phase_ids(progress(env)), step)
                        obs, _, terminated, truncated, _ = env.step(labels)
                        collector.finish(terminated | truncated, env._box_terminal_progress[:, 3])
                    if collector.done:
                        break
                if not collector.done:
                    raise RuntimeError("Reference collection did not complete all first episodes")
                for episode in collector.episodes:
                    episode["batch"] = batch
                episodes.extend(collector.episodes)
                successes = sum(e["success"] for e in collector.episodes)
                print(f"Reference batch{batch}: {successes}/{env.num_envs} successful", flush=True)
        finally:
            env.close()
    bundle = dict(
        version=1,
        population=args.population,
        episodes=episodes,
        teacher_state_dict=teacher_state,
        teacher_sha256=teacher_fingerprint(teacher_state),
        provenance={
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "population": args.population,
            "seed": args.seed,
            "stride": args.stride,
            "episodes": args.episodes,
            "initial_poses": starts,
            "reset_parameters": cfg.events.reset_home.params,
            "presets": ["newton_mjwarp", "newton_renderer"],
        },
    )
    torch.save(bundle, args.output)
    print(
        json.dumps({"output": str(args.output), "successful_episodes": sum(e["success"] for e in episodes)}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
