"""Reuse validated original batches, rerun supplements with corrected provenance."""
import hashlib
import json
import shutil
from pathlib import Path

from isaaclab_tutorial.tasks.place_box import workspace_audit

root = Path(__file__).resolve().parents[1]
parent = root / "logs/workspace_5381_acceptance_20261027_122310"
source = parent / "random"
output = parent / "random_recovered"
original = json.loads((source / "manifest.json").read_text())
run_play = workspace_audit.subprocess.run
reused = []


def reuse_or_run(command, *, env, stdout, stderr, check):
    for job in original["jobs"][:3]:
        if command != job["command"]:
            continue
        name = f"group{job['seed_group']}_batch{job['batch']}"
        result = source / f"{name}.json"
        workspace_audit.read_result(result, job)
        shutil.copy2(result, env["SO101_EVALUATION_OUTPUT"])
        stdout.write((source / f"{name}.log").read_text())
        reused.append(dict(source=str(result), sha256=hashlib.sha256(result.read_bytes()).hexdigest()))
        (output / "reused_batches.json").write_text(json.dumps(reused, indent=2) + "\n")
        print(f"Reused validated original result: {name}", flush=True)
        return
    return run_play(command, env=env, stdout=stdout, stderr=stderr, check=check)


workspace_audit.subprocess.run = reuse_or_run
raise SystemExit(workspace_audit.main([
    "--checkpoint", original["checkpoint"], "--policy", "teacher", "--population", "random",
    "--stage", "F", "--batch-size", "1024", "--output-dir", str(output),
]))
