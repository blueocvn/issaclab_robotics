"""Durable user-authorized overnight training; power control belongs to the Windows wrapper."""

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

from .workspace import STAGES


def latest_checkpoint(directory):
    paths = [p for p in Path(directory).glob("model_*.pt") if re.fullmatch(r"model_\d+\.pt", p.name)]
    return max(paths, key=lambda p: int(p.stem.split("_")[-1])) if paths else None


def usable_checkpoint(directory):
    """An interrupted newest save must not hide earlier readable, finite models."""
    import torch

    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return True

    candidates = [p for p in Path(directory).glob("model_*.pt") if re.fullmatch(r"model_\d+\.pt", p.name)]
    for path in sorted(candidates, key=lambda p: int(p.stem.split("_")[-1]), reverse=True):
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            if ("actor_state_dict" in data or "student_state_dict" in data) and finite(data):
                return path.resolve()
        except Exception:
            continue
    return None


def stage_ready(result, stage):
    if result.get("status") != "pass" or result.get("success_rate", 0) < (0.99 if stage == "F" else 0.95):
        return False
    if stage != "F":
        return True
    if not result.get("spatial_counts") or not result.get("yaw_counts"):
        return False
    pairs = [
        (win, count)
        for a, b in zip(result["spatial_successes"], result["spatial_counts"], strict=True)
        for win, count in zip(a, b, strict=True)
    ]
    pairs += list(zip(result["yaw_successes"], result["yaw_counts"], strict=True))
    return all(count and win / count >= 0.95 for win, count in pairs)


def training_command(policy, checkpoint, stage, run_name, updates, num_envs, fresh=True, learning_rate=None):
    width, yaw = STAGES[stage]
    camera = policy == "camera"
    command = [
        str(Path(sys.executable).with_name("isaaclab")),
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        "IsaacTutorial-Pick-Box-SO101" + ("-Camera" if camera else ""),
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(updates),
        "--seed",
        "42",
        "--run_name",
        run_name,
        "--device",
        "cuda:0",
        "--checkpoint",
        str(checkpoint),
        "--visualizer",
        "none",
        "presets=newton_mjwarp" + (",newton_renderer" if camera else ""),
    ]
    # PPO teacher -> new student uses RSL's ordinary initialization path. A student
    # continuation explicitly loads student+frozen teacher with a fresh lower-LR optimizer.
    if (camera and not fresh) or (not camera and fresh):
        command.append("--no-resume_optimizer_state")
    lr = learning_rate or (0.0005 if camera and fresh else (0.0001 if camera else 0.0003))
    command.extend(
        [
            f"agent.algorithm.learning_rate={lr}",
            f"env.events.reset_home.params.position_noise={width}",
            f"env.events.reset_home.params.yaw_noise={yaw}",
            "env.events.reset_home.params.nominal_fraction=0.2",
        ]
    )
    return command


class Pipeline:
    def __init__(self, output, teacher, max_blocks=8, camera_blocks=3, max_hours=10):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.teacher = Path(teacher).resolve()
        self.max_blocks = max_blocks
        self.camera_blocks = camera_blocks
        self.deadline = time.monotonic() + max_hours * 3600
        self.prefix = self.output.name
        self.state = dict(
            outcome="running",
            phase="initializing",
            started_at=datetime.now(UTC).isoformat(),
            last_checkpoint=str(self.teacher),
            accepted_teacher=None,
            accepted_camera=None,
            max_blocks_per_stage=max_blocks,
            max_camera_blocks=camera_blocks,
            max_hours=max_hours,
        )
        self.record("initialized")

    def save(self):
        self.state["updated_at"] = datetime.now(UTC).isoformat()
        path = self.output / "status.tmp"
        with path.open("w") as stream:
            json.dump(self.state, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        path.replace(self.output / "status.json")
        lines = [
            "# Overnight box training",
            "",
            f"Outcome: **{self.state['outcome']}**",
            f"Phase: {self.state['phase']}",
            f"Updated (UTC): {self.state['updated_at']}",
            "",
            f"Latest checkpoint: `{self.state.get('last_checkpoint')}`",
            f"Accepted wider teacher: `{self.state.get('accepted_teacher')}`",
            f"Accepted camera student: `{self.state.get('accepted_camera')}`",
            "",
            f"Error: {self.state.get('error', 'None')}",
            "",
            "See status.json, journal.jsonl and the training/audit logs in this folder.",
            "A failed outcome does not establish task acceptance. Checkpoints are retained.",
        ]
        (self.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")

    def record(self, event, **fields):
        self.state.update(fields)
        entry = dict(time=datetime.now(UTC).isoformat(), event=event, **fields)
        with (self.output / "journal.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.save()
        print(json.dumps(entry), flush=True)

    def execute(self, command, log):
        if time.monotonic() >= self.deadline:
            raise RuntimeError("Overnight time limit reached; checkpoints and logs preserved")
        if (self.output / "STOP").exists():
            raise RuntimeError("STOP requested; checkpoints and logs preserved")
        self.record("process_started", command=command, current_log=str(log))
        with Path(log).open("w") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.state["child_pid"] = child.pid
            self.save()
            try:
                while child.poll() is None:
                    if time.monotonic() >= self.deadline or (self.output / "STOP").exists():
                        raise RuntimeError("Time limit or STOP reached; preserved the most recent saved checkpoint")
                    self.save()
                    time.sleep(5)
                return child.returncode
            finally:
                # The leader may have exited while an audit/play descendant still runs.
                # Every child starts its own session, so this group belongs only to us.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGTERM)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    child.wait(timeout=30)
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                self.state.pop("child_pid", None)
                self.save()

    def find_run_checkpoint(self, experiment, run_name):
        folders = list((Path("logs/rsl_rl") / experiment).glob(f"*_{run_name}"))
        return usable_checkpoint(folders[0]) if len(folders) == 1 else None

    def train(self, policy, checkpoint, stage, label, updates, fresh=True, learning_rate=None):
        envs = 128 if policy == "camera" else 512
        experiment = "so101_box_camera_distillation" if policy == "camera" else "so101_box_state"
        for attempt in range(3):
            run_name = f"{self.prefix}_{label}_try{attempt}"
            command = training_command(policy, checkpoint, stage, run_name, updates, envs, fresh, learning_rate)
            log = self.output / f"{label}_train_try{attempt}.log"
            try:
                code = self.execute(command, log)
            except Exception:
                saved = self.find_run_checkpoint(experiment, run_name)
                if saved is not None:
                    self.record("partial_checkpoint_recovered", last_checkpoint=str(saved))
                raise
            saved = self.find_run_checkpoint(experiment, run_name)
            if code == 0 and saved is not None:
                self.record("training_completed", last_checkpoint=str(saved.resolve()), training_run=str(saved.parent))
                return saved.resolve()
            text = log.read_text(errors="replace").lower()
            if saved is not None:
                checkpoint = saved.resolve()
                if policy == "camera":
                    fresh = False
                self.record("partial_checkpoint_recovered", last_checkpoint=str(checkpoint))
            if "out of memory" in text or "cuda_error_out_of_memory" in text:
                envs = max(32, envs // 2)
            self.record("execution_retry", retry=attempt + 1, num_envs=envs, failed_log=str(log))
        raise RuntimeError(f"Training execution failed after three attempts: {label}; see {log}")

    def audit(self, policy, checkpoint, population, stage, label):
        for attempt, size in enumerate((1024, 512)):
            directory = self.output / f"{label}_{population}_try{attempt}"
            command = [
                sys.executable,
                "-m",
                "isaaclab_tutorial.tasks.place_box.workspace_audit",
                "--checkpoint",
                str(checkpoint),
                "--policy",
                policy,
                "--population",
                population,
                "--stage",
                stage,
                "--batch-size",
                str(size),
                "--output-dir",
                str(directory),
            ]
            if population == "stage":
                command.extend(["--seeds", "43"])
            log = self.output / f"{label}_{population}_try{attempt}_runner.log"
            code = self.execute(command, log)
            summary = directory / "summary.json"
            if code in (0, 2) and summary.is_file():
                result = json.loads(summary.read_text())
                self.record("audit_completed", audit_directory=str(directory), last_audit=result)
                return result
            self.record("audit_execution_retry", failed_log=str(log), retry=attempt + 1)
        raise RuntimeError(f"Audit execution failed after two attempts: {label}/{population}; see {log}")

    def final_acceptance(self, policy, checkpoint, label):
        results = {
            population: self.audit(policy, checkpoint, population, "F", label)
            for population in ("grid", "random", "nominal", "regression")
        }
        self.record(
            "final_validation",
            validation_policy=policy,
            validation_status={key: value["status"] for key, value in results.items()},
        )
        return all(result["status"] == "pass" for result in results.values())

    def run(self):
        try:
            checkpoint = self.teacher
            for stage in STAGES:
                best, best_score, stale = checkpoint, -1.0, 0
                for block in range(self.max_blocks):
                    label = f"ppo_{stage}_block{block + 1}"
                    self.record("phase", phase=label, stage=stage, block=block + 1)
                    recovering = stale >= 3
                    if recovering:
                        checkpoint = best
                        stale = 0
                    checkpoint = self.train(
                        "teacher",
                        checkpoint,
                        stage,
                        label,
                        200,
                        fresh=block == 0 or recovering,
                        learning_rate=0.0001 if recovering else None,
                    )
                    result = self.audit("teacher", checkpoint, "stage", stage, label)
                    score = result.get("success_rate", 0)
                    if score > best_score:
                        best, best_score, stale = checkpoint, score, 0
                    else:
                        stale += 1
                    if stage_ready(result, stage) and (
                        stage != "F" or self.final_acceptance("teacher", checkpoint, label)
                    ):
                        self.record("stage_accepted", accepted_stage=stage, last_checkpoint=str(checkpoint))
                        break
                else:
                    raise RuntimeError(
                        f"PPO stage {stage} did not pass within {self.max_blocks} training blocks; "
                        f"best stage success={best_score:.4f}, best checkpoint={best}"
                    )
            self.record("teacher_accepted", accepted_teacher=str(checkpoint))
            for block in range(self.camera_blocks):
                label = f"camera_block{block + 1}"
                self.record("phase", phase=label)
                checkpoint = self.train("camera", checkpoint, "F", label, 1600 if block == 0 else 400, fresh=block == 0)
                result = self.audit("camera", checkpoint, "stage", "F", label)
                if stage_ready(result, "F") and self.final_acceptance("camera", checkpoint, label):
                    self.record("completed", outcome="completed", phase="complete", accepted_camera=str(checkpoint))
                    return 0
            raise RuntimeError(f"Camera student did not pass within {self.camera_blocks} training blocks")
        except Exception as error:
            (self.output / "failure_traceback.txt").write_text(traceback.format_exc())
            self.record("failed", outcome="failed", error=str(error))
            return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--max-blocks", type=int, default=8)
    parser.add_argument("--camera-blocks", type=int, default=3)
    parser.add_argument("--max-hours", type=float, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.teacher.is_file() or min(args.max_blocks, args.camera_blocks, args.max_hours) <= 0:
        parser.error("Starting teacher must exist and execution limits must be positive")
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    teacher=str(args.teacher.resolve()),
                    output_dir=str(args.output_dir.resolve()),
                    stages=STAGES,
                    max_blocks_per_stage=args.max_blocks,
                    camera_blocks=args.camera_blocks,
                    max_hours=args.max_hours,
                    shutdown="Windows wrapper only",
                ),
                indent=2,
            )
        )
        return 0
    return Pipeline(args.output_dir, args.teacher, args.max_blocks, args.camera_blocks, args.max_hours).run()


if __name__ == "__main__":
    raise SystemExit(main())
