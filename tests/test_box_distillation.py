"""Data collection, replay and resume contracts without a simulator."""

from types import SimpleNamespace

import pytest
import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from isaaclab_tutorial.tasks.place_box.distillation import DataFirstDistillation, PhaseReplay, phase_ids


def observations(n=4):
    return TensorDict(
        {
            "wrist_rgb": torch.rand(n, 6, 4, 4) - 0.025,
            "proprioception": torch.rand(n, 24),
            "teacher_state": torch.ones(n, 54),
        },
        [n],
    )


class Model(torch.nn.Module):
    is_recurrent = False

    def __init__(self, value):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))
        self.normalized = 0

    def forward(self, obs, **kwargs):
        return self.value.expand(len(obs), 6)

    def reset(self, *args, **kwargs):
        pass

    def update_normalization(self, obs):
        self.normalized += len(obs)


def algorithm(n=4, steps=2, **kwargs):
    obs = observations(n)
    storage = RolloutStorage("distillation", n, steps, obs, [6])
    alg = DataFirstDistillation(
        Model(-2), Model(2), storage, num_learning_epochs=1, gradient_length=1, replay_capacity_per_phase=8, **kwargs
    )
    alg.phase_provider = lambda: torch.arange(n) % 4
    alg.success_provider = lambda: torch.ones(n, dtype=torch.bool)
    return alg, obs


def test_phase_priority_and_detached_snapshot():
    p = SimpleNamespace(
        grasped=torch.tensor([0, 1, 1, 1]).bool(),
        lifted=torch.tensor([0, 0, 1, 1]).bool(),
        placed=torch.tensor([0, 0, 0, 1]).bool(),
    )
    phases = phase_ids(p)
    p.placed[:] = True
    assert phases.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("update", "expected"),
    [(0, 1), (49, 1), (50, 0.5), (149, 0.5), (150, 0.25), (249, 0.25), (250, 0.1), (349, 0.1), (350, 0), (399, 0)],
)
def test_relative_teacher_schedule(update, expected):
    assert DataFirstDistillation.teacher_probability(update) == expected


def test_episode_controller_stable_until_done_and_actions_bounded():
    alg, obs = algorithm()
    assert torch.equal(alg.act(obs), torch.ones(4, 6))
    alg.num_updates = 350
    alg.process_env_step(obs, torch.zeros(4), torch.tensor([1, 0, 1, 0]), {})
    assert alg.teacher_mask.tolist() == [False, True, False, True]
    assert alg.act(obs)[:, 0].tolist() == [-1, 1, -1, 1]
    assert alg.storage.privileged_actions[0].max() == 1
    assert not any(p.requires_grad for p in alg.teacher.parameters())


def test_preaction_phase_and_observations_are_copied_before_env_mutation():
    alg, obs = algorithm()
    original = obs.clone()
    phase = torch.arange(4)
    alg.phase_provider = lambda: phase
    alg.act(obs)
    obs["wrist_rgb"].fill_(99)
    phase.fill_(3)
    alg.process_env_step(obs, torch.zeros(4), torch.zeros(4), {})
    assert torch.equal(alg.rollout_phases[0], torch.arange(4))
    assert torch.equal(alg.storage.observations[0]["wrist_rgb"], original["wrist_rgb"])


def test_replay_balances_available_phases_and_excludes_privileged_inputs():
    replay = PhaseReplay(3, seed=8)
    obs = observations(12)
    obs["wrist_rgb"][0].fill_(-0.025)
    replay.add(obs, torch.ones(12, 6), torch.arange(12) % 4)
    batch, labels, phases = replay.sample(64, "cpu")
    assert phases.bincount(minlength=4).tolist() == [16] * 4
    assert set(batch.keys()) == {"wrist_rgb", "proprioception"}
    assert labels.shape == (64, 6)
    assert replay.sizes == [3] * 4
    assert replay.buckets[0]["wrist_rgb"].dtype == torch.float16
    assert (batch["wrist_rgb"] < 0).any()


def test_replay_ring_is_bounded_and_empty_buckets_redistributed():
    replay = PhaseReplay(2, seed=3)
    assert replay.sample(64, "cpu") is None
    obs = observations(5)
    obs["proprioception"][:, 0] = torch.arange(5)
    replay.add(obs, torch.ones(5, 6), torch.zeros(5, dtype=torch.long))
    batch, _, phases = replay.sample(64, "cpu")
    assert replay.sizes == [2, 0, 0, 0]
    assert set(batch["proprioception"][:, 0].tolist()) == {3, 4}
    assert phases.tolist() == [0] * 64


def test_replay_snapshot_roundtrip_and_private_rng_do_not_touch_global_rng():
    replay = PhaseReplay(8, seed=17)
    obs = observations(12)
    replay.add(obs, torch.ones(12, 6), torch.arange(12) % 4)
    state = replay.state_dict()
    restored = PhaseReplay(8, seed=99)
    restored.load_state_dict(state)
    global_rng = torch.get_rng_state().clone()
    a = replay.sample(64, "cpu")
    b = restored.sample(64, "cpu")
    assert torch.equal(a[0]["wrist_rgb"], b[0]["wrist_rgb"])
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert torch.allclose(a[0]["wrist_rgb"].float(), a[0]["wrist_rgb"].half().float(), atol=0)
    assert (obs["wrist_rgb"].half().float() - obs["wrist_rgb"]).abs().max() < 0.001


def collect(alg, obs):
    for _ in range(alg.storage.num_transitions_per_env):
        alg.act(obs)
        alg.process_env_step(obs, torch.zeros(len(obs)), torch.zeros(len(obs)), {})


def test_update_uses_earlier_replay_only_and_normalizes_fresh_only():
    alg, obs = algorithm()
    collect(alg, obs)
    first = alg.update()
    assert first["replay_samples"] == 0
    assert sum(alg.replay.sizes) == 8
    assert alg.student.normalized == 8
    collect(alg, obs)
    second = alg.update()
    assert second["replay_samples"] == 128  # two optimizer steps, 64 replay each
    assert alg.student.normalized == 16
    assert alg.optimizer.state[alg.student.value]["step"] == 4
    assert alg.num_updates == 2
    assert alg.storage.step == 0
    assert alg.teacher.value.item() == 2


def test_checkpoint_restores_replay_schedule_rng_and_legacy_starts_empty():
    alg, obs = algorithm()
    collect(alg, obs)
    alg.update()
    alg.num_updates = 160
    saved = alg.save()
    clone, _ = algorithm()
    assert clone.load(saved, None, True)
    assert clone.num_updates == 160
    assert clone.replay.sizes == alg.replay.sizes
    assert torch.equal(clone.controller_rng.get_state(), alg.controller_rng.get_state())
    assert clone.teacher_mask is None  # simulator episodes are reset on resume
    legacy = {k: v for k, v in saved.items() if k != "box_data_first"}
    clone.load(legacy, None, True)
    assert clone.num_updates == 0
    assert sum(clone.replay.sizes) == 0


def test_inference_never_uses_teacher_even_during_teacher_only_seeding():
    alg, obs = algorithm()
    alg.eval_mode()
    assert alg.get_policy() is alg.student
    assert alg.get_policy()(obs)[0, 0].item() == -2
    assert alg.act(obs)[0, 0].item() == -1


def test_student_episode_metrics_exclude_teacher_controlled_episodes():
    alg, obs = algorithm()
    alg.act(obs)
    alg.teacher_mask = torch.tensor([True, False, True, False])
    alg.success_provider = lambda: torch.tensor([True, False, False, True])
    alg.process_env_step(obs, torch.zeros(4), torch.ones(4), {})
    alg.act(obs)
    alg.process_env_step(obs, torch.zeros(4), torch.zeros(4), {})
    metrics = alg.update()
    assert metrics["student_episodes"] == 2
    assert metrics["student_successes"] == 1
    assert metrics["teacher_episodes"] == 2
    assert metrics["teacher_successes"] == 1


def test_opt_in_entry_point_preserves_default_box_and_vial_trainers():
    import gymnasium as gym

    import isaaclab_tutorial.tasks  # noqa: F401
    from isaaclab_tutorial.tasks.place_box.agents import BoxDataFirstDistillationRunnerCfg

    kwargs = gym.spec("IsaacTutorial-Pick-Box-SO101-Camera").kwargs
    assert kwargs["rsl_rl_data_first_cfg_entry_point"].endswith(":BoxDataFirstDistillationRunnerCfg")
    assert kwargs["rsl_rl_cfg_entry_point"].endswith(":BoxDistillationRunnerCfg")
    assert "rsl_rl_data_first_cfg_entry_point" not in gym.spec("IsaacTutorial-Place-Vial-SO101-Camera").kwargs
    cfg = BoxDataFirstDistillationRunnerCfg()
    assert cfg.init_at_random_ep_len is False
    assert cfg.algorithm.learning_rate == 0.0001
