"""Run the handoff's evaluation-only workflow and preserve each population."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[1]
checkpoint = root / "logs/rsl_rl/so101_box_state/2026-10-27_03-19-50_so101-box-workspace-F-refine/model_5381.pt"
output = root / "logs" / ("workspace_5381_acceptance_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
output.mkdir(exist_ok=False)
print(f"OUTPUT={output}", flush=True)
results = {}
for population in (sys.argv[1:] or ("stage", "grid", "random", "nominal", "regression")):
    command = [sys.executable, "-m", "isaaclab_tutorial.tasks.place_box.workspace_audit",
               "--checkpoint", str(checkpoint), "--policy", "teacher",
               "--population", population, "--stage", "F", "--batch-size", "1024",
               "--output-dir", str(output / population)]
    if population == "stage":
        command += ["--seeds", "43", "45"]
    print(f"START {population}", flush=True)
    with (output / f"{population}_runner.log").open("w") as stream:
        result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    summary_path = output / population / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    results[population] = dict(returncode=result.returncode, summary=summary)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"END {population}: exit={result.returncode}; "
          f"result={None if summary is None else (summary['successes'], summary['episodes'], summary['status'])}", flush=True)
    if result.returncode not in (0, 2):
        raise SystemExit(f"Execution error; inspect {output / (population + '_runner.log')}")
print(f"FINISHED={output}", flush=True)
