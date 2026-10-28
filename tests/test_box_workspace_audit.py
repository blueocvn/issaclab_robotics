"""Coverage and promotion rules for reproducible workspace audits."""

import json
import math

import pytest
import torch

from isaaclab_tutorial.tasks.place_box.workspace import grid_offsets


def report(rows, population="grid", policy="teacher"):
    from isaaclab_tutorial.tasks.place_box.workspace_report import summarize

    return summarize(rows, population, 0.1, math.pi, policy)


def grid_rows(repeats=15):
    rows = []
    for i, (x, y, yaw) in enumerate(grid_offsets(torch.arange(200 * repeats), 0.1, math.pi).tolist()):
        rows.append(
            dict(
                initial_box_pose=[0.17 + x, -0.1 + y, 0.0805, 0, 0, math.sin(yaw / 2), math.cos(yaw / 2)],
                success=True,
                grasped=True,
                lifted=True,
                reset_source="grid",
                case_id=i % 1000,
                seed_group=45 + i // 1000,
            )
        )
    return rows


def test_full_grid_passes_with_all_boundary_cases_and_sufficient_counts():
    result = report(grid_rows())
    assert result["status"] == "pass"
    assert result["episodes"] == 3000
    assert result["spatial_counts"] == [[120] * 5] * 5
    assert result["yaw_counts"] == [375] * 8
    assert len(result["case_counts"]) == 200
    assert 0 < result["success_wilson_95"][0] < 1


def test_missing_grid_case_and_duplicate_cannot_pass_even_with_perfect_aggregate():
    rows = grid_rows()
    rows[0] = rows[1].copy()
    assert report(rows)["status"] == "insufficient_coverage"


def test_small_smoke_grid_is_explicitly_insufficient():
    assert report(grid_rows(1))["status"] == "insufficient_coverage"


def test_nominal_training_contamination_is_rejected():
    rows = grid_rows()
    rows[10]["reset_source"] = "nominal"
    with pytest.raises(ValueError, match="source"):
        report(rows)


def test_failure_cluster_blocks_promotion_despite_high_overall_score():
    rows = grid_rows()
    # Seven failures in one spatial point:99.77% aggregate but94.17% in that point.
    indices = [i for i in range(len(rows)) if i % 200 < 8][:7]
    for i in indices:
        rows[i]["success"] = False
    result = report(rows)
    assert result["success_rate"] > 0.99
    assert result["status"] == "fail"


def test_nominal_policy_specific_thresholds_and_regression_failure():
    rows = [dict(success=i >= 4, reset_source="current", seed_group=42 + i // 1024) for i in range(3072)]
    assert report(rows, "nominal", "teacher")["status"] == "fail"
    assert report(rows, "nominal", "camera")["status"] == "pass"
    regression = [dict(success=True, reset_source="replay") for _ in range(3)]
    assert report(regression, "regression")["status"] == "pass"
    regression[0]["success"] = False
    assert report(regression, "regression")["status"] == "fail"


def test_repeated_single_seed_group_cannot_pass_final_coverage():
    rows = grid_rows()
    for row in rows:
        row["seed_group"] = 45
    assert report(rows)["status"] == "insufficient_coverage"


def test_grid_case_labels_cannot_hide_wrong_physical_start():
    rows = grid_rows()
    rows[0]["initial_box_pose"] = rows[1]["initial_box_pose"].copy()
    with pytest.raises(ValueError, match="grid"):
        report(rows)


def test_batched_jobs_cover_every_grid_index_without_repeating_random_streams(tmp_path):
    from isaaclab_tutorial.tasks.place_box.workspace_audit import build_jobs

    jobs = build_jobs(tmp_path / "model.pt", "teacher", "grid", "F", [45], 256)
    indices = [i for job in jobs for i in range(job["start_index"], job["start_index"] + job["count"])]
    assert indices == list(range(1000))
    assert len({job["sampling_seed"] for job in jobs}) == len(jobs)
    assert all("env.events.reset_home.params.nominal_fraction=0.0" in job["command"] for job in jobs)


def test_balanced_duplicate_grid_repeat_ids_cannot_replace_missing_repeats():
    rows = grid_rows()
    for row in rows:
        row["case_id"] %= 200
    assert report(rows)["status"] == "insufficient_coverage"


def test_truncated_planned_audit_cannot_report_pass():
    from isaaclab_tutorial.tasks.place_box.workspace_audit import finalize_report

    result = finalize_report({"status": "pass", "coverage_sufficient": True}, 1, 12)
    assert result["status"] == "insufficient_coverage"
    assert not result["coverage_sufficient"]


def test_overlapping_seed_groups_cannot_duplicate_actual_sampling_streams(tmp_path):
    from isaaclab_tutorial.tasks.place_box.workspace_audit import build_jobs

    jobs = build_jobs(tmp_path / "model.pt", "teacher", "grid", "F", [45, 1000048, 2000051], 256)
    assert len({job["sampling_seed"] for job in jobs}) == len(jobs)


@pytest.mark.parametrize("corruption", ["case", "seed"])
def test_batch_results_must_match_planned_case_indices_and_sampling_seed(tmp_path, corruption):
    from isaaclab_tutorial.tasks.place_box.workspace_audit import build_jobs, read_result

    job = build_jobs(tmp_path / "model.pt", "teacher", "grid", "F", [45], 2)[1]
    data = dict(
        argv=job["command"][1:],
        summary={"episodes": 2, "success_rate": 1.0},
        reset_parameters=dict(
            nominal_fraction=0,
            reset_mode="grid",
            grid_start_index=job["start_index"],
            sampling_seed=job["sampling_seed"],
        ),
        episodes=[dict(env_id=i, success=True, case_id=job["start_index"] + i) for i in range(2)],
    )
    if corruption == "case":
        data["episodes"][0]["case_id"] = 0
    else:
        data["reset_parameters"]["sampling_seed"] += 1
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_result(path, job)


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"grid_repeats": 0}, {"seeds": []}])
def test_invalid_job_dimensions_are_rejected(tmp_path, kwargs):
    from isaaclab_tutorial.tasks.place_box.workspace_audit import build_jobs

    args = dict(
        checkpoint=tmp_path / "model.pt", policy="teacher", population="grid", stage="F", seeds=[45], batch_size=200
    )
    args.update(kwargs)
    with pytest.raises(ValueError):
        build_jobs(**args)


def test_random_supplement_preserves_provenance_through_result_validation(tmp_path, monkeypatch):
    from pathlib import Path

    from isaaclab_tutorial.tasks.place_box import workspace_audit

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    output = tmp_path / "audit"

    def simulate_play(command, *, env, stdout, stderr, check):
        overrides = dict(item.split("=", 1) for item in command if item.startswith("env.events."))
        size = int(command[command.index("--num_envs") + 1])
        prefix = "env.events.reset_home.params."
        data = dict(
            argv=command[1:],
            summary={"episodes": size, "success_rate": 1.0},
            reset_parameters=dict(
                nominal_fraction=0,
                reset_mode=overrides[prefix + "reset_mode"],
                grid_start_index=int(overrides[prefix + "grid_start_index"]),
                sampling_seed=int(overrides[prefix + "sampling_seed"]),
            ),
            episodes=[dict(env_id=i, success=True) for i in range(size)],
        )
        Path(env["SO101_EVALUATION_OUTPUT"]).write_text(json.dumps(data))

    def coverage_report(rows, *args, **kwargs):
        count = 100 if len(rows) > 3 else 99
        return dict(
            status="pass",
            coverage_sufficient=True,
            episodes=len(rows),
            spatial_counts=[[count] * 5 for _ in range(5)],
            yaw_counts=[100] * 8,
        )

    monkeypatch.setattr(workspace_audit.subprocess, "run", simulate_play)
    monkeypatch.setattr(workspace_audit, "summarize", coverage_report)
    assert (
        workspace_audit.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--policy",
                "teacher",
                "--population",
                "random",
                "--episodes",
                "1",
                "--batch-size",
                "1",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed_jobs"] == summary["planned_jobs"] == 4
    assert summary["episodes"] == 259


def test_custom_episode_horizon_changes_only_requested_audit_override():
    from isaaclab_tutorial.tasks.place_box.workspace_audit import build_jobs

    kwargs = dict(
        checkpoint="model.pt",
        policy="camera",
        population="nominal",
        stage="F",
        seeds=[42],
        batch_size=256,
        episodes=256,
        agent="rsl_rl_camera_ppo_cfg_entry_point",
    )
    normal = build_jobs(**kwargs)[0]["command"]
    longer = build_jobs(**kwargs, episode_length_s=60.0)[0]["command"]
    assert longer[:-1] == normal
    assert longer[-1] == "env.episode_length_s=60.0"


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_audit_episode_horizon_is_rejected(duration):
    from isaaclab_tutorial.tasks.place_box.workspace_audit import build_jobs

    with pytest.raises(ValueError, match="duration"):
        build_jobs("model.pt", "camera", "nominal", "F", [42], 256, episode_length_s=duration)
