"""Paired experimental improvement gates are distinct from deployment gates."""

import json

import pytest

from isaaclab_tutorial.tasks.place_box.camera_comparison import (
    compare_candidates,
    load_population,
    main,
    paired_summary,
)


def rows(successes):
    return [
        dict(seed_group=42, batch=0, env_id=i, initial_box_pose=[i, 0, 0, 0, 0, 0, 1], success=s)
        for i, s in enumerate(successes)
    ]


def test_paired_summary_counts_bootstrap_and_repeatability():
    baseline = rows([False] * 10)
    candidate = rows([True] * 10)
    summary = paired_summary(baseline, candidate)
    assert summary["gain_pp"] == 100
    assert summary["gain_paired_bootstrap_95_pp"] == [100, 100]
    assert summary["candidate_only_successes"] == 10
    assert paired_summary(baseline, candidate) == summary
    assert paired_summary(candidate, candidate)["gain_paired_bootstrap_95_pp"] == [0, 0]


def test_pairing_rejects_pose_changes_duplicates_and_missing_rows():
    baseline = rows([True, False])
    with pytest.raises(ValueError, match="identities"):
        paired_summary(baseline, baseline[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        paired_summary(baseline, [baseline[0], baseline[0]])
    candidate = rows([False, False])
    candidate[0]["initial_box_pose"][0] = 99
    with pytest.raises(ValueError, match="poses"):
        paired_summary(baseline, candidate)


def test_prefer_control_without_resolved_new_trainer_advantage():
    baseline = {"nominal": rows([True] * 100), "wide": rows([True] * 20 + [False] * 80)}
    arm = {"nominal": rows([True] * 98 + [False] * 2), "wide": rows([True] * 40 + [False] * 60)}
    report = compare_candidates(baseline, {"control": arm, "data-first": arm})
    assert report["selected"] == "control"
    assert report["candidates"]["data-first"]["qualifies"]


def test_failed_nominal_gate_retains_baseline_and_resolved_gain_selects_new():
    baseline = {"nominal": rows([True] * 100), "wide": rows([True] * 20 + [False] * 80)}
    failed = {"nominal": rows([True] * 96 + [False] * 4), "wide": rows([True] * 90 + [False] * 10)}
    assert compare_candidates(baseline, {"data-first": failed})["selected"] == "baseline"
    control = {"nominal": baseline["nominal"], "wide": rows([True] * 40 + [False] * 60)}
    better = {"nominal": baseline["nominal"], "wide": rows([True] * 70 + [False] * 30)}
    assert compare_candidates(baseline, {"control": control, "data-first": better})["selected"] == "data-first"


def audit(directory, population, checkpoint, digest):
    from isaaclab_tutorial.tasks.place_box.workspace_audit import build_jobs

    directory.mkdir()
    seed = 52 if population == "nominal" else 49
    job = build_jobs(checkpoint, "camera", population, "F", [seed], 2, episodes=2)[0]
    manifest = dict(
        policy="camera",
        population=population,
        stage="F",
        jobs=[job],
        completed_jobs=1,
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=digest,
    )
    (directory / "manifest.json").write_text(json.dumps(manifest))
    data = dict(
        episodes=rows([True, False]),
        summary=dict(episodes=2, success_rate=0.5),
        reset_parameters=dict(nominal_fraction=0, sampling_seed=seed, grid_start_index=0, reset_mode="random"),
        argv=job["command"][1:],
    )
    (directory / f"group{seed}_batch0.json").write_text(json.dumps(data))
    return directory


@pytest.mark.parametrize("mixed", ["baseline", "candidate"])
def test_cli_rejects_mixed_nominal_wide_checkpoint_identities(tmp_path, mixed):
    base_n = audit(tmp_path / "base_n", "nominal", tmp_path / "base.pt", "a" * 64)
    base_w = audit(
        tmp_path / "base_w",
        "stage",
        tmp_path / "other.pt" if mixed == "baseline" else tmp_path / "base.pt",
        ("c" if mixed == "baseline" else "a") * 64,
    )
    cand_n = audit(tmp_path / "cand_n", "nominal", tmp_path / "candidate.pt", "b" * 64)
    cand_w = audit(
        tmp_path / "cand_w",
        "stage",
        tmp_path / "other.pt" if mixed == "candidate" else tmp_path / "candidate.pt",
        ("c" if mixed == "candidate" else "b") * 64,
    )
    with pytest.raises(ValueError, match="same checkpoint"):
        main(
            [
                "--baseline-nominal",
                str(base_n),
                "--baseline-wide",
                str(base_w),
                "--candidate",
                "control",
                str(cand_n),
                str(cand_w),
                "--episodes",
                "2",
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
    assert not (tmp_path / "result.json").exists()


def test_loader_rejects_job_checkpoint_disagreeing_with_manifest(tmp_path):
    directory = audit(tmp_path / "audit", "nominal", tmp_path / "base.pt", "a" * 64)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["checkpoint"] = str(tmp_path / "different.pt")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest checkpoint"):
        load_population(directory, "nominal", 2)
