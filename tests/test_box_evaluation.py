"""A fast environment cannot contribute more than its first episode to the audit."""

import json
from types import SimpleNamespace

import pytest
import torch

from isaaclab_rl import rsl_rl

from isaaclab_tutorial.tasks.place_box import evaluation


class FakeWrapper:
    def __init__(self):
        self.num_envs = 2
        self.unwrapped = self
        self.device = "cpu"
        self._box_initial_pose = torch.zeros(2, 7)
        self._box_terminal_progress = torch.tensor([[True, True, True, True], [True, False, False, False]])
        self._box_terminal_details = {
            "on_target": torch.tensor([True, True]),
            "released": torch.tensor([True, False]),
            "withdrawn": torch.tensor([True, False]),
            "upright": torch.tensor([True, True]),
        }
        self.termination_manager = SimpleNamespace(get_term=lambda name: torch.tensor([False, name == "time_out"]))
        self.dones = iter((torch.tensor([True, False]), torch.tensor([True, False]), torch.tensor([True, True])))

    def step(self, actions):
        return None, None, next(self.dones), {}


def test_box_audit_counts_first_episode_only_and_saves_failure_modes(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rsl_rl, "RslRlVecEnvWrapper", FakeWrapper)
    monkeypatch.setattr(FakeWrapper, "step", FakeWrapper.step)
    monkeypatch.setattr(evaluation, "EXACT_EVALUATION_ACTIVE", False)
    monkeypatch.setenv("SO101_EVALUATION_EPISODES", "2")
    output = tmp_path / "box_audit.json"
    monkeypatch.setenv("SO101_EVALUATION_OUTPUT", str(output))
    evaluation.install_episode_counter()
    wrapper = FakeWrapper()
    wrapper.step(None)
    wrapper.step(None)
    with pytest.raises(SystemExit) as result:
        wrapper.step(None)
    assert result.value.code == 0
    saved = json.loads(output.read_text())
    assert saved["summary"]["episodes"] == 2
    assert saved["summary"]["success_rate"] == 0.5
    assert saved["summary"]["timed_out_rate"] == 0.5
    assert saved["summary"]["on_target_rate"] == 1.0
    assert saved["summary"]["released_rate"] == 0.5
    assert saved["summary"]["withdrawn_rate"] == 0.5
    assert saved["episodes"][1]["on_target"] and not saved["episodes"][1]["released"]
    assert len(saved["episodes"]) == 2
    assert "SO101_BOX_EVAL_RESULT=" in capsys.readouterr().out


def test_box_audit_preserves_start_provenance_before_auto_reset(monkeypatch, tmp_path):
    class ResettingWrapper(FakeWrapper):
        def __init__(self):
            super().__init__()
            self._box_reset_source = torch.tensor([2, 2])
            self._box_reset_case = torch.tensor([200, 201])
            self._box_reset_parameters = {"reset_mode": "grid", "nominal_fraction": 0}

        def step(self, actions):
            self._box_reset_source[:] = 0
            self._box_reset_case[:] = -1
            return super().step(actions)

    monkeypatch.setattr(rsl_rl, "RslRlVecEnvWrapper", ResettingWrapper)
    monkeypatch.setattr(ResettingWrapper, "step", ResettingWrapper.step)
    monkeypatch.setattr(evaluation, "EXACT_EVALUATION_ACTIVE", False)
    monkeypatch.setenv("SO101_EVALUATION_EPISODES", "2")
    output = tmp_path / "provenance.json"
    monkeypatch.setenv("SO101_EVALUATION_OUTPUT", str(output))
    evaluation.install_episode_counter()
    wrapper = ResettingWrapper()
    wrapper.step(None)
    wrapper.step(None)
    with pytest.raises(SystemExit):
        wrapper.step(None)
    data = json.loads(output.read_text())
    assert data["reset_parameters"]["reset_mode"] == "grid"
    assert [r["reset_source"] for r in data["episodes"]] == ["grid", "grid"]
    assert [r["case_id"] for r in data["episodes"]] == [200, 201]
