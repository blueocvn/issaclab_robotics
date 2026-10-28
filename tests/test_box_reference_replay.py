"""Protected-reference retention, provenance and trainer integration contracts."""

import copy

import pytest
import torch
from rsl_rl.storage import RolloutStorage
from test_box_distillation import Model, collect, observations

from isaaclab_tutorial.tasks.place_box.reference_replay import (
    AnchoredDistillation,
    ReferenceBank,
    build_reference,
    teacher_fingerprint,
)


def episodes():
    result = []
    for population, marker in (("nominal", 10), ("wide", 20)):
        obs = observations(16)
        obs["proprioception"].fill_(marker)
        result.append(
            dict(
                population=population,
                success=True,
                observations=obs,
                labels=torch.ones(16, 6),
                phases=torch.arange(16) % 4,
            )
        )
    return result


def payload():
    return build_reference(episodes(), Model(2).state_dict(), capacity_per_phase=4, seed=0)


def anchored(reference=None, **kwargs):
    obs = observations()
    obs["proprioception"].fill_(40)
    storage = RolloutStorage("distillation", 4, 2, obs, [6])
    alg = AnchoredDistillation(
        Model(-2),
        Model(2),
        storage,
        num_learning_epochs=1,
        gradient_length=1,
        replay_capacity_per_phase=8,
        reference_bank=reference,
        **kwargs,
    )
    alg.phase_provider = lambda: torch.arange(4)
    alg.success_provider = lambda: torch.ones(4, dtype=torch.bool)
    return alg, obs


def test_reference_rejects_missing_population_or_phase_and_failed_episodes():
    rows = episodes()
    rows[1]["success"] = False
    with pytest.raises(ValueError, match="population|phase"):
        build_reference(rows, Model(2).state_dict())
    rows = episodes()
    rows[1]["phases"].fill_(0)
    with pytest.raises(ValueError, match="phase"):
        build_reference(rows, Model(2).state_dict())


def test_failed_examples_are_excluded_and_sampling_balances_populations_and_phases():
    rows = episodes()
    failed = copy.deepcopy(rows[0])
    failed["success"] = False
    failed["observations"]["proprioception"].fill_(99)
    bank = ReferenceBank(build_reference(rows + [failed], Model(2).state_dict(), capacity_per_phase=2))
    batch, labels, phases = bank.sample(32, "cpu")
    assert batch["proprioception"][:, 0].tolist().count(10) == 16
    assert batch["proprioception"][:, 0].tolist().count(20) == 16
    assert phases.bincount(minlength=4).tolist() == [8] * 4
    assert set(batch.keys()) == {"wrist_rgb", "proprioception"}
    assert torch.equal(labels, torch.ones(32, 6))


def test_reference_owns_data_and_checkpoint_restores_private_sampling_rng():
    data = payload()
    bank = ReferenceBank(data)
    for ring in data["populations"].values():
        ring["buckets"][0]["labels"].fill_(99)
    state = bank.state_dict()
    restored = ReferenceBank(state)
    rng = torch.get_rng_state().clone()
    a = bank.sample(32, "cpu")
    b = restored.sample(32, "cpu")
    assert torch.equal(a[0]["wrist_rgb"], b[0]["wrist_rgb"])
    assert a[1].max() == 1
    assert torch.equal(rng, torch.get_rng_state())


def test_trainer_requires_reference_and_matching_teacher_before_collection():
    alg, obs = anchored()
    with pytest.raises(ValueError, match="reference"):
        alg.load(
            {
                "student_state_dict": alg.student.state_dict(),
                "teacher_state_dict": alg.teacher.state_dict(),
                "optimizer_state_dict": alg.optimizer.state_dict(),
            },
            None,
            True,
        )
    alg, obs = anchored(payload())
    saved = alg.save()
    saved["teacher_state_dict"] = Model(3).state_dict()
    with pytest.raises(ValueError, match="teacher"):
        alg.load(saved, None, True)


def test_updates_mix_reference_with_fresh_and_rolling_without_mutating_bank():
    alg, obs = anchored(payload())
    original = alg.reference.state_dict()
    batches = []
    alg.student.register_forward_pre_hook(
        lambda module, args: batches.append(args[0]["proprioception"][:, 0].tolist()) if len(args[0]) == 128 else None
    )
    obs["proprioception"].fill_(30)
    collect(alg, obs)
    first = alg.update()
    assert first["reference_samples"] == 64  # 32 for each of two optimizer steps
    assert first["rolling_samples"] == 0
    assert first["fresh_samples"] == 192
    assert batches[-1].count(30) == 96
    assert batches[-1].count(10) == 16
    assert batches[-1].count(20) == 16
    obs["proprioception"].fill_(40)
    collect(alg, obs)
    second = alg.update()
    assert second["reference_samples"] == 64
    assert second["rolling_samples"] == 64
    assert second["fresh_samples"] == 128
    assert batches[-1].count(40) == 64
    assert batches[-1].count(30) == 32
    assert batches[-1].count(10) == 16
    assert batches[-1].count(20) == 16
    assert alg.student.normalized == 16
    for pop in ("nominal", "wide"):
        for before, after in zip(
            original["populations"][pop]["buckets"],
            alg.reference.state_dict()["populations"][pop]["buckets"],
            strict=True,
        ):
            for key in before:
                assert torch.equal(before[key], after[key])


def test_anchored_checkpoint_resumes_without_external_bank_and_resets_legacy_bank():
    alg, obs = anchored(payload())
    collect(alg, obs)
    alg.update()
    saved = copy.deepcopy(alg.save())
    clone, _ = anchored()
    clone.load(saved, None, True)
    assert clone.num_updates == 1
    assert clone.replay.sizes == alg.replay.sizes
    assert teacher_fingerprint(clone.teacher.state_dict()) == saved["box_reference"]["teacher_sha256"]
    a = alg.reference.sample(32, "cpu")
    b = clone.reference.sample(32, "cpu")
    assert torch.equal(a[0]["wrist_rgb"], b[0]["wrist_rgb"])
    legacy = {k: v for k, v in saved.items() if k not in ("box_reference", "box_data_first")}
    with pytest.raises(ValueError, match="reference"):
        clone.load(legacy, None, True)


def test_nonfinite_or_out_of_bounds_reference_is_rejected():
    rows = episodes()
    rows[0]["labels"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        build_reference(rows, Model(2).state_dict())
    rows[0]["labels"].fill_(2)
    with pytest.raises(ValueError):
        build_reference(rows, Model(2).state_dict())


def test_first_episode_collector_copies_preaction_data_and_ignores_later_episodes():
    from isaaclab_tutorial.tasks.place_box.reference_collection import FirstEpisodeReferences

    collector = FirstEpisodeReferences(2, "nominal", stride=1)
    obs = observations(2)
    obs["proprioception"].fill_(7)
    collector.capture(obs, torch.ones(2, 6), torch.tensor([0, 1]), 0)
    obs["proprioception"].fill_(99)
    collector.finish(torch.tensor([True, False]), torch.tensor([True, False]))
    collector.capture(obs, torch.ones(2, 6), torch.tensor([2, 3]), 1)
    collector.finish(torch.tensor([True, True]), torch.tensor([False, False]))
    assert collector.done
    assert len(collector.episodes) == 2
    good = next(e for e in collector.episodes if e["success"])
    assert good["observations"]["proprioception"][:, 0].tolist() == [7]
    assert good["phases"].tolist() == [0]


def test_merge_rejects_different_teachers_and_keeps_collection_provenance():
    from isaaclab_tutorial.tasks.place_box.reference_collection import merge_collections

    teacher = Model(2).state_dict()
    nominal = dict(
        population="nominal",
        teacher_state_dict=teacher,
        teacher_sha256=teacher_fingerprint(teacher),
        episodes=episodes()[:1],
        provenance={"seed": 43},
    )
    wide = dict(
        population="wide",
        teacher_state_dict=teacher,
        teacher_sha256=teacher_fingerprint(teacher),
        episodes=episodes()[1:],
        provenance={"seed": 44},
    )
    merged = merge_collections(nominal, wide, capacity_per_phase=4)
    assert ReferenceBank(merged).state_dict()["collection_provenance"] == [{"seed": 43}, {"seed": 44}]
    wide["teacher_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="teacher"):
        merge_collections(nominal, wide)


def test_anchored_registration_does_not_replace_previous_trainers():
    import gymnasium as gym

    import isaaclab_tutorial.tasks  # noqa: F401
    from isaaclab_tutorial.tasks.place_box.agents import BoxAnchoredDistillationRunnerCfg

    kwargs = gym.spec("IsaacTutorial-Pick-Box-SO101-Camera").kwargs
    assert kwargs["rsl_rl_anchored_cfg_entry_point"].endswith(":BoxAnchoredDistillationRunnerCfg")
    assert kwargs["rsl_rl_data_first_cfg_entry_point"].endswith(":BoxDataFirstDistillationRunnerCfg")
    assert kwargs["rsl_rl_cfg_entry_point"].endswith(":BoxDistillationRunnerCfg")
    assert "rsl_rl_anchored_cfg_entry_point" not in gym.spec("IsaacTutorial-Place-Vial-SO101-Camera").kwargs
    cfg = BoxAnchoredDistillationRunnerCfg()
    assert cfg.algorithm.reference_path is None


def test_resume_embedded_bank_ignores_missing_original_path_in_saved_config(tmp_path):
    path = tmp_path / "reference.pt"
    torch.save(payload(), path)
    alg, obs = anchored(reference_path=path)
    collect(alg, obs)
    alg.update()
    saved = copy.deepcopy(alg.save())
    path.unlink()
    resumed, _ = anchored(reference_path=path)
    resumed.load(saved, None, True)
    assert resumed.num_updates == 1
    assert resumed.reference.sample(32, "cpu")[1].max() == 1
