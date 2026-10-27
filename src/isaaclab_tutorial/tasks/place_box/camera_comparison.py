"""Compare matched student-only camera audits with the experimental gain rule.

This does not change workspace_report's deployment acceptance thresholds.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def paired_summary(baseline, candidate):
    def indexed(rows):
        result = {(row["seed_group"], row["batch"], row["env_id"]): row for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate paired episode identities")
        return result

    a, b = indexed(baseline), indexed(candidate)
    if not a or a.keys() != b.keys():
        raise ValueError("Paired episode identities differ or are empty")
    for key in a:
        if a[key]["initial_box_pose"] != b[key]["initial_box_pose"]:
            raise ValueError("Paired initial poses differ")
        if not isinstance(a[key]["success"], bool) or not isinstance(b[key]["success"], bool):
            raise ValueError("Success outcomes must be booleans")
    wins = sum(b[key]["success"] and not a[key]["success"] for key in a)
    losses = sum(a[key]["success"] and not b[key]["success"] for key in a)
    n = len(a)
    # Resampling the three paired outcomes is exactly the paired bootstrap of
    # binary success differences, without allocating10000 full episode arrays.
    draws = np.random.default_rng(0).multinomial(n, [wins / n, losses / n, (n - wins - losses) / n], size=10_000)
    intervals = np.quantile(100 * (draws[:, 0] - draws[:, 1]) / n, [0.025, 0.975]).tolist()
    return {
        "episodes": n,
        "baseline_successes": sum(row["success"] for row in a.values()),
        "candidate_successes": sum(row["success"] for row in b.values()),
        "candidate_only_successes": wins,
        "baseline_only_successes": losses,
        "both_successes": sum(a[key]["success"] and b[key]["success"] for key in a),
        "both_failures": sum(not a[key]["success"] and not b[key]["success"] for key in a),
        "gain_pp": 100 * (wins - losses) / n,
        "gain_paired_bootstrap_95_pp": intervals,
    }


def compare_candidates(baseline, candidates):
    if not candidates or set(candidates) - {"control", "data-first"}:
        raise ValueError("Candidates must be control and/or data-first")
    report = {"bootstrap_resamples": 10_000, "bootstrap_seed": 0, "selected": "baseline", "candidates": {}}
    for name, populations in candidates.items():
        result = {pop: paired_summary(baseline[pop], populations[pop]) for pop in ("nominal", "wide")}
        result["qualifies"] = result["nominal"]["gain_pp"] >= -3 and result["wide"]["gain_pp"] >= 10
        report["candidates"][name] = result
    qualifying = [name for name, result in report["candidates"].items() if result["qualifies"]]
    if len(qualifying) == 1:
        report["selected"] = qualifying[0]
    elif len(qualifying) == 2:
        head_to_head = paired_summary(candidates["control"]["wide"], candidates["data-first"]["wide"])
        report["data_first_vs_control_wide"] = head_to_head
        report["selected"] = "data-first" if head_to_head["gain_paired_bootstrap_95_pp"][0] > 0 else "control"
    report["interpretation"] = (
        "Experimental improvement selection, not deployment acceptance. "
        "Intervals are descriptive paired bootstrap intervals; no automatic training extension."
    )
    return report


def load_population(directory, population, episodes):
    from .workspace_audit import read_result

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    jobs = manifest["jobs"]
    if (
        manifest["policy"] != "camera"
        or manifest["population"] != population
        or manifest.get("completed_jobs") != len(jobs)
        or not jobs
        or (population == "stage" and manifest["stage"] != "F")
    ):
        raise ValueError(f"Incomplete or wrong camera audit: {directory}")
    digest = manifest.get("checkpoint_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise ValueError(f"Missing/invalid checkpoint SHA256: {directory}")
    rows, protocol = [], []
    for job in jobs:
        result = directory / f"group{job['seed_group']}_batch{job['batch']}.json"
        rows.extend(read_result(result, job))
        command = job["command"].copy()
        checkpoint_index = command.index("--checkpoint") + 1
        if Path(command[checkpoint_index]).resolve() != Path(manifest["checkpoint"]).resolve():
            raise ValueError(f"Audit job disagrees with manifest checkpoint: {directory}")
        command[checkpoint_index] = "CHECKPOINT"
        protocol.append(command)
    if len(rows) != episodes:
        raise ValueError(f"Expected{episodes} first episodes in {directory}, found{len(rows)}")
    return rows, protocol, digest.lower()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-nominal", type=Path, required=True)
    parser.add_argument("--baseline-wide", type=Path, required=True)
    parser.add_argument(
        "--candidate", nargs=3, action="append", metavar=("NAME", "NOMINAL_DIR", "WIDE_DIR"), required=True
    )
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.output.exists():
        parser.error("Episodes must be positive and the output must be a new file")
    baseline, protocols, identities = {}, {}, {}
    for key, pop, directory in (("nominal", "nominal", args.baseline_nominal), ("wide", "stage", args.baseline_wide)):
        baseline[key], protocols[key], identities[key] = load_population(directory, pop, args.episodes)
    if identities["nominal"] != identities["wide"]:
        raise ValueError("Baseline nominal and wide audits must use the same checkpoint")
    checkpoint_hashes = {"baseline": identities["nominal"]}
    candidates = {}
    for name, nominal_dir, wide_dir in args.candidate:
        if name in candidates:
            parser.error(f"Duplicate candidate: {name}")
        candidates[name] = {}
        candidate_hashes = {}
        for key, pop, directory in (("nominal", "nominal", nominal_dir), ("wide", "stage", wide_dir)):
            rows, protocol, candidate_hashes[key] = load_population(directory, pop, args.episodes)
            if protocol != protocols[key]:
                raise ValueError("Audit commands differ beyond the checkpoint; comparison is not controlled")
            candidates[name][key] = rows
        if candidate_hashes["nominal"] != candidate_hashes["wide"]:
            raise ValueError(f"Candidate{name} nominal and wide audits must use the same checkpoint")
        checkpoint_hashes[name] = candidate_hashes["nominal"]
    report = compare_candidates(baseline, candidates)
    report["checkpoint_sha256"] = checkpoint_hashes
    report["sources"] = {
        "baseline_nominal": str(args.baseline_nominal),
        "baseline_wide": str(args.baseline_wide),
        "candidates": args.candidate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
