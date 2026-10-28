"""Protected successful references alongside rolling phase replay (opt-in)."""

import copy
import hashlib
from pathlib import Path

import torch

from .distillation import STUDENT_KEYS, DataFirstDistillation, PhaseReplay


def teacher_fingerprint(state):
    """Stable identity of teacher tensors, independent of checkpoint packaging."""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_rows(observations, labels, phases):
    n = len(labels)
    if (
        n == 0
        or labels.shape != (n, 6)
        or phases.shape != (n,)
        or phases.dtype != torch.long
        or ((phases < 0) | (phases > 3)).any()
    ):
        raise ValueError("Invalid reference labels/phases")
    if labels.abs().max() > 1 or not torch.isfinite(labels).all():
        raise ValueError("Reference labels must be finite and bounded")
    for key, ndim in (("wrist_rgb", 4), ("proprioception", 2)):
        value = observations[key]
        if value.ndim != ndim or len(value) != n or not torch.isfinite(value).all():
            raise ValueError(f"Invalid reference observation: {key}")


def build_reference(episodes, teacher_state, capacity_per_phase=1024, seed=43):
    """Freeze a phase/population-balanced sample of successful teacher episodes."""
    populations, counts = {}, {}
    rng = torch.Generator().manual_seed(seed)
    for position, population in enumerate(("nominal", "wide")):
        good = [e for e in episodes if e["population"] == population and e["success"]]
        if not good:
            raise ValueError(f"No successful reference population: {population}")
        observations = {key: torch.cat([e["observations"][key].cpu() for e in good]) for key in STUDENT_KEYS}
        labels = torch.cat([e["labels"].cpu() for e in good])
        phases = torch.cat([e["phases"].cpu() for e in good])
        _validate_rows(observations, labels, phases)
        if (phases.bincount(minlength=4) == 0).any():
            raise ValueError(f"Missing reference phase in {population}")
        order = torch.randperm(len(labels), generator=rng)
        ring = PhaseReplay(capacity_per_phase, seed + position)
        ring.add({key: value[order] for key, value in observations.items()}, labels[order], phases[order])
        populations[population] = ring.state_dict()
        counts[population] = len(good)
    return {
        "version": 1,
        "teacher_sha256": teacher_fingerprint(teacher_state),
        "populations": populations,
        "successful_episodes": counts,
    }


class ReferenceBank:
    """Immutable examples; only the private sampling RNGs advance."""

    def __init__(self, state):
        if state.get("version") != 1 or set(state.get("populations", {})) != {"nominal", "wide"}:
            raise ValueError("Unsupported reference version/populations")
        fingerprint = state.get("teacher_sha256", "")
        if len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
            raise ValueError("Invalid reference teacher fingerprint")
        self.teacher_sha256 = fingerprint
        self.successful_episodes = dict(state.get("successful_episodes", {}))
        self.collection_provenance = copy.deepcopy(state.get("collection_provenance", []))
        self._populations = {}
        for population, saved in state["populations"].items():
            if len(saved["sizes"]) != 4 or min(saved["sizes"]) <= 0:
                raise ValueError("Reference requires all four phases")
            for phase, (size, bucket) in enumerate(zip(saved["sizes"], saved["buckets"], strict=True)):
                if set(bucket) != {*STUDENT_KEYS, "labels"}:
                    raise ValueError("Invalid reference observation keys")
                _validate_rows(bucket, bucket["labels"], torch.full((size,), phase, dtype=torch.long))
            ring = PhaseReplay(saved["capacity"])
            ring.load_state_dict(saved)
            self._populations[population] = ring

    def sample(self, count, device):
        if count <= 0 or count % 8:
            raise ValueError("Reference sample count must be a positive multiple of eight")
        chunks = [self._populations[p].sample(count // 2, device) for p in ("nominal", "wide")]
        return tuple(torch.cat([chunk[i] for chunk in chunks], dim=0) for i in range(3))

    def state_dict(self):
        return {
            "version": 1,
            "teacher_sha256": self.teacher_sha256,
            "successful_episodes": self.successful_episodes.copy(),
            "collection_provenance": copy.deepcopy(self.collection_provenance),
            "populations": {key: value.state_dict() for key, value in self._populations.items()},
        }


class AnchoredDistillation(DataFirstDistillation):
    """64 fresh +32 rolling +32 protected references per minibatch."""

    def __init__(self, *args, reference_path=None, reference_bank=None, **kwargs):
        super().__init__(*args, **kwargs)
        if reference_path and reference_bank is not None:
            raise ValueError("Specify one reference source")
        self._reference_path = Path(reference_path) if reference_path else None
        self._reference_source = reference_bank
        # A resume checkpoint may embed the bank even if its original path is gone.
        self.reference = ReferenceBank(reference_bank) if reference_bank is not None else None
        self._reference_verified = False
        self._reference_samples = self._rolling_samples = self._fresh_samples = 0

    def _initial_reference(self):
        data = self._reference_source
        if self._reference_path is not None:
            data = torch.load(self._reference_path, map_location="cpu", weights_only=False)
        return ReferenceBank(data) if data is not None else None

    def _verify_reference(self):
        if self.reference is None:
            self.reference = self._initial_reference()
        if self.reference is None:
            raise ValueError("A protected reference bank is required before training")
        if self.reference.teacher_sha256 != teacher_fingerprint(self.teacher.state_dict()):
            raise ValueError("Reference teacher does not match the loaded teacher")
        self._reference_verified = True

    def act(self, obs):
        if self.collecting and not self._reference_verified:
            self._verify_reference()
        return super().act(obs)

    def sample_retained(self):
        if not self._reference_verified:
            self._verify_reference()
        reference = self.reference.sample(32, self.device)
        rolling = self.replay.sample(32, self.device)
        self._reference_samples += 32
        self._rolling_samples += 32 if rolling is not None else 0
        self._fresh_samples += 64 if rolling is not None else 96
        if rolling is None:
            return reference
        return tuple(torch.cat((rolling[i], reference[i]), dim=0) for i in range(3))

    def update(self):
        self._reference_samples = self._rolling_samples = self._fresh_samples = 0
        metrics = super().update()
        metrics.update(
            reference_samples=self._reference_samples,
            rolling_samples=self._rolling_samples,
            fresh_samples=self._fresh_samples,
        )
        return metrics

    def save(self):
        saved = super().save()
        if self.reference is not None:
            saved["box_reference"] = self.reference.state_dict()
        return saved

    def load(self, loaded_dict, load_cfg, strict):
        result = super().load(loaded_dict, load_cfg, strict)
        restore_student = load_cfg is None or load_cfg.get("student", False)
        saved = loaded_dict.get("box_reference") if restore_student else None
        self.reference = ReferenceBank(saved) if saved is not None else self._initial_reference()
        self._reference_verified = False
        self._verify_reference()
        return result
