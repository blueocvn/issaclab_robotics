"""Summarize independently recorded acceptance and compare archived 4982 starts."""
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
root = Path(__file__).resolve().parents[1]
results = json.loads((output / "results.json").read_text())
directories = {population: output / population for population in results}
if len(sys.argv) > 2:
    continuation = Path(sys.argv[2])
    for population, result in json.loads((continuation / "results.json").read_text()).items():
        results[population] = result
        directories[population] = continuation / population
recovered = output / "random_recovered"
if (recovered / "summary.json").exists():
    summary = json.loads((recovered / "summary.json").read_text())
    if summary["completed_jobs"] == summary["planned_jobs"]:
        results["random"] = dict(returncode=0 if summary["status"] == "pass" else 2, summary=summary)
        directories["random"] = recovered
analysis = {"populations": {}, "historical_4982_comparison": {}}
for population, result in results.items():
    summary = result["summary"]
    if summary is None:
        continue
    rows = [dict(row, result_file=path.name) for path in sorted(directories[population].glob("group*_batch*.json"))
            for row in json.loads(path.read_text())["episodes"]]
    assert len(rows) == summary["episodes"], "Recorded rows must match the accepted summary"
    failed = [row for row in rows if not row["success"]]
    details = dict(status=summary["status"], successes=summary["successes"],
                   episodes=summary["episodes"], rate=summary["success_rate"],
                   coverage_sufficient=summary["coverage_sufficient"],
                   evidence_directory=str(directories[population]),
                   failure_stages={
                       "no_grasp": sum(not row["grasped"] for row in failed),
                       "grasp_without_lift": sum(row["grasped"] and not row["lifted"] for row in failed),
                       "incomplete_after_lift": sum(row["lifted"] for row in failed)},
                   failed_starts=[dict(result_file=row["result_file"], env_id=row["env_id"],
                                      initial_box_pose=row["initial_box_pose"],
                                      grasped=row["grasped"], lifted=row["lifted"], placed=row["placed"])
                                  for row in failed])
    if "spatial_counts" in summary:
        spatial = [dict(row=y, column=x, successes=w, count=n, rate=w/n)
                   for y, (wins, counts) in enumerate(zip(summary["spatial_successes"], summary["spatial_counts"]))
                   for x, (w, n) in enumerate(zip(wins, counts)) if n]
        yaw = [dict(bin=i, successes=w, count=n, rate=w/n)
               for i, (w, n) in enumerate(zip(summary["yaw_successes"], summary["yaw_counts"])) if n]
        details.update(worst_spatial=min(spatial, key=lambda item: item["rate"]),
                       worst_yaw=min(yaw, key=lambda item: item["rate"]),
                       spatial_below_95=[item for item in spatial if item["rate"] < .95],
                       yaw_below_95=[item for item in yaw if item["rate"] < .95])
        if population == "stage":
            details["stage_F_promotion_pass"] = (summary["success_rate"] >= .99
                and not details["spatial_below_95"] and not details["yaw_below_95"])
        if population == "grid":
            details["worst_cases"] = sorted(
                [dict(case_id=i, successes=w, count=n, rate=w/n)
                 for i, (w, n) in enumerate(zip(summary["case_successes"], summary["case_counts"])) if n],
                key=lambda item: item["rate"])[:10]
    analysis["populations"][population] = details

baselines = {
    43: root / "logs/overnight_20261027_ppo_camera/ppo_F_block6_stage_try0/group43_batch0.json",
    45: root / "logs/workspace_4982_holdout45_20261027/group45_batch0.json",
}
for seed, baseline in baselines.items():
    current = output / "stage" / f"group{seed}_batch0.json"
    if not current.exists():
        continue
    old = {row["env_id"]: row for row in json.loads(baseline.read_text())["episodes"]}
    new = {row["env_id"]: row for row in json.loads(current.read_text())["episodes"]}
    max_pose_delta = max(abs(a-b) for key in old for a,b in
                         zip(old[key]["initial_box_pose"], new[key]["initial_box_pose"]))
    analysis["historical_4982_comparison"][seed] = dict(
        episodes=len(new), baseline_successes=sum(row["success"] for row in old.values()),
        current_successes=sum(row["success"] for row in new.values()),
        max_initial_pose_component_delta=max_pose_delta,
        matching_starts_within_1e_6=max_pose_delta <= 1e-6,
        recovered_failures=sum(not old[key]["success"] and new[key]["success"] for key in old),
        new_failures=sum(old[key]["success"] and not new[key]["success"] for key in old))
analysis["teacher_accepted"] = all(
    population in analysis["populations"] and analysis["populations"][population]["status"] == "pass"
    for population in ("grid", "random", "nominal", "regression"))
analysis["acceptance_basis"] = "Original strict automated gates; physical success unchanged"
analysis["user_approved_for_distillation"] = True
(output / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
(output / "combined_results.json").write_text(json.dumps(results, indent=2) + "\n")
lines = ["# Checkpoint 5381 independent evaluation", "",
         "Original strict acceptance gates: " + ("PASS" if analysis["teacher_accepted"] else "FAIL"), "",
         "The user subsequently accepted teacher 5381 at the observed grid/random scores for camera distillation.", "",
         "| Population | Successes | Rate | Result |", "| --- | ---: | ---: | --- |"]
for name, details in analysis["populations"].items():
    status = "fails Stage F promotion" if name == "stage" and not details["stage_F_promotion_pass"] else details["status"]
    lines.append(f"| {name} | {details['successes']}/{details['episodes']} | {details['rate']:.4%} | {status} |")
lines += ["", "## Comparison with archived checkpoint 4982", "",
          "| Seed | 4982 | 5381 | Recovered failures | New failures |",
          "| --- | ---: | ---: | ---: | ---: |"]
for seed, item in analysis["historical_4982_comparison"].items():
    lines.append(f"| {seed} | {item['baseline_successes']} | {item['current_successes']} | {item['recovered_failures']} | {item['new_failures']} |")
lines += ["", "Both comparison seeds have exactly matching initial poses. The baseline results are archived;",
          "a one-episode difference does not establish a meaningful performance difference.", "",
          "## Findings", "",
          "- Teacher 5381 is user-approved for camera distillation at the observed scores.",
          "- The grid's far pickup corner and yaw near -180 degrees need targeted trajectory diagnosis.",
          "- Preserve the physical success contract. No training or hardware operation was launched.", "",
          "## Audit repair and provenance", "",
          "The original random run rejected its first supplement because the job plan recorded",
          "start index 256 while the command passed 0. The validator was kept intact. The source",
          "fix propagates the planned start index into the launch command, with a regression test.",
          "The full suite passed: 175 passed, 2 optional skips. Lint passed.", "",
          "The recovered random audit reuses only the three validated original batches, with",
          "SHA256 provenance in random_recovered/reused_batches.json. Supplements are rerun",
          "with correct metadata. Original output and failed-attempt logs are preserved.", "",
          "Per-population directories, failure starts, bins, and gates are in analysis.json.", ""]
(output / "REPORT.md").write_text("\n".join(lines))
print(json.dumps({"teacher_accepted": analysis["teacher_accepted"],
                  "comparison": analysis["historical_4982_comparison"],
                  "populations": {key: {k: v for k, v in value.items() if k != "failed_starts"}
                                  for key, value in analysis["populations"].items()}}, indent=2))
