"""Overnight orchestration must preserve gating and recoverable checkpoints."""

import contextlib
import json
import os
import signal
import sys
from pathlib import Path

import pytest


def pipeline():
    from isaaclab_tutorial.tasks.place_box import workspace_pipeline

    return workspace_pipeline


def test_checkpoint_selection_uses_numeric_iteration(tmp_path):
    for name in ("model_99.pt", "model_100.pt", "model_9.pt", "other.pt"):
        (tmp_path / name).touch()
    assert pipeline().latest_checkpoint(tmp_path).name == "model_100.pt"


def test_stage_readiness_never_accepts_incomplete_or_failed_audits():
    w = pipeline()
    assert not w.stage_ready(dict(status="insufficient_coverage", success_rate=1.0), "A")
    assert w.stage_ready(dict(status="pass", success_rate=0.96), "A")
    assert not w.stage_ready(dict(status="pass", success_rate=0.98), "F")


def test_full_yaw_stage_requires_good_spatial_and_yaw_bins():
    result = dict(
        status="pass",
        success_rate=0.995,
        spatial_counts=[[100] * 5] * 5,
        spatial_successes=[[100] * 5] * 5,
        yaw_counts=[100] * 8,
        yaw_successes=[100] * 8,
    )
    assert pipeline().stage_ready(result, "F")
    result["yaw_successes"][0] = 90
    assert not pipeline().stage_ready(result, "F")


def test_camera_initialization_and_continuation_use_correct_load_mode():
    w = pipeline()
    first = w.training_command("camera", "teacher.pt", "F", "first", 1600, 128, fresh=True)
    assert "--no-resume_optimizer_state" not in first
    later = w.training_command("camera", "student.pt", "F", "later", 400, 128, fresh=False)
    assert "--no-resume_optimizer_state" in later
    assert "agent.algorithm.learning_rate=0.0001" in later
    ppo = w.training_command("teacher", "teacher.pt", "A", "ppo", 200, 512, fresh=True)
    assert "--no-resume_optimizer_state" in ppo
    assert "env.events.reset_home.params.nominal_fraction=0.2" in ppo


def test_execution_failure_is_saved_and_blocks_camera_training(tmp_path, monkeypatch):
    w = pipeline()
    controller = w.Pipeline(tmp_path / "run", tmp_path / "teacher.pt", max_blocks=1)
    calls = []

    def failed_train(policy, *args, **kwargs):
        calls.append(policy)
        raise RuntimeError("unrecoverable test failure")

    monkeypatch.setattr(controller, "train", failed_train)
    assert controller.run() == 1
    state = json.loads((controller.output / "status.json").read_text())
    assert state["outcome"] == "failed"
    assert "unrecoverable test failure" in state["error"]
    assert calls == ["teacher"]
    assert (controller.output / "SUMMARY.md").exists()


def test_unaccepted_teacher_never_reaches_camera_and_retries_are_bounded(tmp_path, monkeypatch):
    w = pipeline()
    controller = w.Pipeline(tmp_path / "run", tmp_path / "teacher.pt", max_blocks=2)
    calls = []
    monkeypatch.setattr(controller, "train", lambda policy, *a, **kw: calls.append(policy) or tmp_path / "new.pt")
    monkeypatch.setattr(controller, "audit", lambda *a, **kw: dict(status="fail", success_rate=0.4))
    assert controller.run() == 1
    assert calls == ["teacher", "teacher"]
    assert "stage A" in json.loads((controller.output / "status.json").read_text())["error"]


def test_camera_requires_all_four_independent_final_populations(tmp_path, monkeypatch):
    controller = pipeline().Pipeline(tmp_path / "run", tmp_path / "teacher.pt")
    populations = []

    def audit(policy, checkpoint, population, stage, label):
        populations.append(population)
        return dict(status="fail" if population == "nominal" else "pass")

    monkeypatch.setattr(controller, "audit", audit)
    assert not controller.final_acceptance("teacher", tmp_path / "model.pt", "final")
    assert populations == ["grid", "random", "nominal", "regression"]


def test_corrupt_newest_checkpoint_does_not_hide_older_valid_progress(tmp_path):
    import torch

    torch.save({"actor_state_dict": {"weight": torch.ones(1)}}, tmp_path / "model_50.pt")
    (tmp_path / "model_100.pt").write_bytes(b"interrupted save")
    assert pipeline().usable_checkpoint(tmp_path).name == "model_50.pt"


def test_deadline_exception_records_newer_saved_checkpoint_before_exit(tmp_path, monkeypatch):
    controller = pipeline().Pipeline(tmp_path / "run", tmp_path / "old.pt")
    newer = tmp_path / "model_50.pt"
    monkeypatch.setattr(controller, "find_run_checkpoint", lambda *a: newer, raising=False)

    def deadline(*args):
        raise RuntimeError("deadline")

    monkeypatch.setattr(controller, "execute", deadline)
    with pytest.raises(RuntimeError, match="deadline"):
        controller.train("teacher", tmp_path / "old.pt", "A", "test", 200)
    assert json.loads((controller.output / "status.json").read_text())["last_checkpoint"] == str(newer)


@pytest.mark.skipif(os.name != "posix", reason="Linux process-group cleanup")
def test_finished_group_leader_cannot_leave_a_running_descendant(tmp_path):
    controller = pipeline().Pipeline(tmp_path / "run", tmp_path / "old.pt")
    log = tmp_path / "child.log"
    code = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(600)']);print(p.pid,flush=True)"
    )
    try:
        assert controller.execute([sys.executable, "-c", code], log) == 0
        descendant = int(log.read_text().strip())
        stat = Path(f"/proc/{descendant}/stat")
        assert not stat.exists() or stat.read_text().split()[2] == "Z"
    finally:
        if log.exists() and log.read_text().strip().isdigit():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(log.read_text().strip()), signal.SIGKILL)
