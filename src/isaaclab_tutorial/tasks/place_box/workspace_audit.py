"""Run controlled box workspace audits in fresh play processes; no training.

Example: python -m isaaclab_tutorial.tasks.place_box.workspace_audit
--checkpoint logs/.../model.pt --policy teacher --population grid --stage F
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .workspace import STAGES
from .workspace_report import summarize


def reserve_seed(group, batch, used):
    seed = (group + batch * 1_000_003) % (2**31)
    while seed in used:
        seed = (seed + 1) % (2**31)
    used.add(seed)
    return seed


def finalize_report(result, completed, planned):
    result = dict(result, completed_jobs=completed, planned_jobs=planned)
    if completed != planned:
        result.update(status="insufficient_coverage", coverage_sufficient=False)
    return result


def build_jobs(checkpoint, policy, population, stage, seeds, batch_size, episodes=1024, grid_repeats=5):
    """Return reproducible play commands; indices remain global across batches."""
    if (
        batch_size <= 0
        or grid_repeats <= 0
        or episodes <= 0
        or not seeds
        or any(not 0 <= seed < 2**31 for seed in seeds)
    ):
        raise ValueError("Audit sizes and seed groups must be positive/nonempty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed groups must be unique")
    if policy not in ("teacher", "camera") or population not in ("grid", "random", "stage", "nominal", "regression"):
        raise ValueError("Unknown policy or population")
    width, yaw = (0.01, 0.1) if population == "nominal" else STAGES[stage]
    count = 200 * grid_repeats if population == "grid" else (3 if population == "regression" else episodes)
    jobs = []
    used = set()
    for group in seeds:
        for batch, start in enumerate(range(0, count, batch_size)):
            size = min(batch_size, count - start)
            seed = reserve_seed(group, batch, used)
            mode = {"grid": "grid", "regression": "replay"}.get(population, "random")
            command = [
                str(Path(sys.executable).with_name("isaaclab")),
                "play",
                "--rl_library",
                "rsl_rl",
                "--task",
                "IsaacTutorial-Pick-Box-SO101" + ("-Camera" if policy == "camera" else ""),
                "--num_envs",
                str(size),
                "--seed",
                str(seed),
                "--device",
                "cuda:0",
                "--checkpoint",
                str(Path(checkpoint).resolve()),
                "--visualizer",
                "none",
                "--external_callback",
                "isaaclab_tutorial.tasks.place_box.evaluation.install_episode_counter",
                "presets=newton_mjwarp" + (",newton_renderer" if policy == "camera" else ""),
            ]
            overrides = dict(
                position_noise=width,
                yaw_noise=yaw,
                nominal_fraction=0.0,
                reset_mode=mode,
                sampling_seed=seed,
                grid_start_index=start,
            )
            command.extend(f"env.events.reset_home.params.{key}={value}" for key, value in overrides.items())
            jobs.append(
                dict(
                    seed_group=group,
                    batch=batch,
                    start_index=start,
                    count=size,
                    sampling_seed=seed,
                    reset_mode=mode,
                    command=command,
                )
            )
    return jobs


def read_result(path, job):
    data = json.loads(path.read_text())
    rows = data["episodes"]
    if (
        len(rows) != job["count"]
        or data["summary"]["episodes"] != job["count"]
        or {row["env_id"] for row in rows} != set(range(job["count"]))
    ):
        raise ValueError(f"Missing/duplicate first episodes in {path}")
    for key, value in data["summary"].items():
        if key.endswith("_rate") and abs(sum(row[key[:-5]] for row in rows) / len(rows) - value) > 1e-12:
            raise ValueError(f"Summary mismatch in {path}: {key}")
    if data["reset_parameters"]["nominal_fraction"] != 0:
        raise ValueError("Training mixture is forbidden in acceptance audits")
    parameters = data["reset_parameters"]
    if (
        parameters["sampling_seed"] != job["sampling_seed"]
        or parameters["grid_start_index"] != job["start_index"]
        or parameters["reset_mode"] != job["reset_mode"]
        or data["argv"] != job["command"][1:]
    ):
        raise ValueError("Returned audit configuration does not match the planned job")
    if job["reset_mode"] in ("grid", "replay") and any(
        row["case_id"] != job["start_index"] + row["env_id"] for row in rows
    ):
        raise ValueError("Returned case indices do not match the planned batch")
    return [dict(row, seed_group=job["seed_group"], batch=job["batch"]) for row in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy", choices=("teacher", "camera"), required=True)
    parser.add_argument("--population", choices=("grid", "random", "stage", "nominal", "regression"), required=True)
    parser.add_argument("--stage", choices=STAGES, default="F")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=1024, help="Random/stage/nominal starts per seed group")
    parser.add_argument("--grid-repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--regression-source", type=Path, default=Path("logs/box_release_v3_1997_evaluation.json"))
    parser.add_argument(
        "--max-jobs", type=int, default=0, help="Bounded integration check; incomplete coverage cannot pass"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file() or args.max_jobs < 0:
        parser.error("Checkpoint must exist and max-jobs must be nonnegative")
    seeds = args.seeds or (
        [42] if args.population == "regression" else ([42, 43, 44] if args.population == "nominal" else [45, 46, 47])
    )
    if args.population == "regression" and len(seeds) != 1:
        parser.error("Regression uses exactly one seed group and three replays")
    try:
        jobs = build_jobs(
            args.checkpoint,
            args.policy,
            args.population,
            args.stage,
            seeds,
            args.batch_size,
            args.episodes,
            args.grid_repeats,
        )
    except ValueError as error:
        parser.error(str(error))
    output = args.output_dir or Path("logs") / (
        "workspace_" + datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.policy}_{args.population}_{args.stage}"
    )
    if args.population == "regression":
        replay = output.resolve() / "case527.json"
        for job in jobs:
            job["command"].append(f"env.events.reset_home.params.replay_path={json.dumps(str(replay))}")
    if args.dry_run:
        print(json.dumps(dict(output_dir=str(output), jobs=jobs), indent=2))
        return 0
    # A fresh directory prevents accidental reuse/overwrite of old model results.
    output.mkdir(parents=True, exist_ok=False)
    if args.population == "regression":
        source = json.loads(args.regression_source.read_text())
        poses = [r["initial_box_pose"] for r in source["episodes"] if r["env_id"] == 527]
        if len(poses) != 1:
            raise ValueError("Regression source must contain exactly one episode527")
        replay.write_text(json.dumps(dict(poses=poses)))
    with args.checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = dict(
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=digest,
        policy=args.policy,
        population=args.population,
        stage=args.stage,
        seed_groups=seeds,
        jobs=jobs,
    )
    rows = []
    width, yaw = (0.01, 0.1) if args.population == "nominal" else STAGES[args.stage]
    completed = 0
    index = 0
    while index < len(jobs):
        if args.max_jobs and completed >= args.max_jobs:
            break
        job = jobs[index]
        manifest["started_job"] = index
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        name = f"group{job['seed_group']}_batch{job['batch']}"
        result_path = output / f"{name}.json"
        env = dict(
            os.environ, SO101_EVALUATION_EPISODES=str(job["count"]), SO101_EVALUATION_OUTPUT=str(result_path.resolve())
        )
        print(f"Audit {index + 1}/{len(jobs)}: {name}, {job['count']} first episodes", flush=True)
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(job["command"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows.extend(read_result(result_path, job))
        completed += 1
        index += 1
        result = summarize(
            rows, args.population, width, yaw, args.policy, expected_per_case=args.grid_repeats * len(seeds)
        )
        # Supplement random coverage only after all initial groups, using predetermined256-case streams.
        if args.population == "random" and index == len(jobs):
            enough = min(min(r) for r in result["spatial_counts"]) >= 100 and min(result["yaw_counts"]) >= 100
            if not enough:
                group = seeds[(index + 1) % len(seeds)]
                batch = 1 + max(j["batch"] for j in jobs if j["seed_group"] == group)
                extra = build_jobs(args.checkpoint, args.policy, "random", args.stage, [group], 256, episodes=256)[0]
                seed = reserve_seed(group, batch, {j["sampling_seed"] for j in jobs})
                extra.update(batch=batch, sampling_seed=seed, start_index=batch * 256)
                command = extra["command"]
                command[command.index("--seed") + 1] = str(seed)
                command[:] = [
                    f"env.events.reset_home.params.sampling_seed={seed}"
                    if x.startswith("env.events.reset_home.params.sampling_seed=")
                    else x
                    for x in command
                ]
                jobs.append(extra)
        result = finalize_report(result, completed, len(jobs))
        manifest["completed_jobs"] = completed
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
