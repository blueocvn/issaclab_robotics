"""Opt-in episode-level teacher collection with phase-balanced retained examples.

Phase/controller metadata is used only for collection and sampling. The student
continues to consume wrist_rgb and proprioception; inference uses its raw policy.
"""

import torch
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict

STUDENT_KEYS = ("wrist_rgb", "proprioception")
PHASE_NAMES = ("approach", "grasp", "lift", "placed")


def phase_ids(progress):
    """Snapshot pre-action progress, with later phases taking precedence."""
    return torch.where(progress.placed, 3, torch.where(progress.lifted, 2, progress.grasped.long())).clone()


class PhaseReplay:
    """Four independent CPU rings; serialize only initialized rows."""

    def __init__(self, capacity_per_phase=4096, seed=44):
        if capacity_per_phase <= 0:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity_per_phase
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.buckets = [{} for _ in PHASE_NAMES]
        self.sizes = [0] * 4
        self.cursors = [0] * 4

    def add(self, observations, labels, phases):
        source = {
            key: observations[key].detach().to("cpu", dtype=(torch.float16 if key == "wrist_rgb" else torch.float32))
            for key in STUDENT_KEYS
        }
        source["labels"] = labels.detach().to("cpu", dtype=torch.float32)
        phases = phases.detach().to("cpu")
        for phase in range(4):
            ids = (phases == phase).nonzero().flatten()
            if not len(ids):
                continue
            bucket = self.buckets[phase]
            if not bucket:
                bucket.update(
                    {
                        key: torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
                        for key, value in source.items()
                    }
                )
            # At most the most recent capacity rows can survive this insertion.
            skip = max(0, len(ids) - self.capacity)
            cursor = (self.cursors[phase] + skip) % self.capacity
            ids = ids[skip:]
            indices = (torch.arange(len(ids)) + cursor) % self.capacity
            for key, value in source.items():
                bucket[key][indices] = value[ids]
            self.cursors[phase] = (cursor + len(ids)) % self.capacity
            self.sizes[phase] = min(self.capacity, self.sizes[phase] + skip + len(ids))

    def sample(self, count, device):
        available = [i for i, size in enumerate(self.sizes) if size]
        if not available:
            return None
        chunks, phase_chunks = [], []
        for position, phase in enumerate(available):
            quota = count // len(available) + (position < count % len(available))
            if not quota:
                continue
            indices = torch.randint(self.sizes[phase], (quota,), generator=self.rng)
            chunks.append({key: value[indices] for key, value in self.buckets[phase].items()})
            phase_chunks.append(torch.full((quota,), phase, dtype=torch.long))
        values = {
            key: torch.cat([chunk[key] for chunk in chunks]).to(device, dtype=torch.float32)
            for key in (*STUDENT_KEYS, "labels")
        }
        return (
            TensorDict({key: values[key] for key in STUDENT_KEYS}, [count], device=device),
            values["labels"],
            torch.cat(phase_chunks),
        )

    def state_dict(self):
        return {
            "version": 1,
            "capacity": self.capacity,
            "sizes": self.sizes.copy(),
            "cursors": self.cursors.copy(),
            "rng": self.rng.get_state().clone(),
            "buckets": [
                {key: value[:size].clone() for key, value in bucket.items()}
                for bucket, size in zip(self.buckets, self.sizes, strict=True)
            ],
        }

    def load_state_dict(self, state):
        if state["version"] != 1 or state["capacity"] != self.capacity:
            raise ValueError("Unsupported replay version/capacity")
        if len(state["sizes"]) != 4 or len(state["cursors"]) != 4 or len(state["buckets"]) != 4:
            raise ValueError("Replay requires four phase rings")
        for size, cursor in zip(state["sizes"], state["cursors"], strict=True):
            if not 0 <= size <= self.capacity or not 0 <= cursor < self.capacity:
                raise ValueError("Invalid replay ring state")
        self.sizes, self.cursors = list(state["sizes"]), list(state["cursors"])
        self.buckets = []
        for size, saved in zip(self.sizes, state["buckets"], strict=True):
            bucket = {}
            for key, value in saved.items():
                if len(value) != size:
                    raise ValueError("Replay rows do not match occupancy")
                bucket[key] = torch.empty((self.capacity, *value.shape[1:]), dtype=value.dtype, device="cpu")
                bucket[key][:size].copy_(value.cpu())
            self.buckets.append(bucket)
        self.rng.set_state(state["rng"].cpu())


class DataFirstDistillation(Distillation):
    """Feedforward box distillation with persistent balanced replay."""

    def __init__(self, *args, replay_capacity_per_phase=4096, data_seed=43, **kwargs):
        super().__init__(*args, **kwargs)
        if self.gradient_length != 1 or self.is_multi_gpu or self.student.is_recurrent:
            raise ValueError("Data-first distillation requires feedforward, single-GPU, gradient_length=1")
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.data_seed = data_seed
        self.controller_rng = torch.Generator(device="cpu").manual_seed(data_seed + 1)
        self.fresh_rng = torch.Generator(device="cpu").manual_seed(data_seed + 2)
        self.replay = PhaseReplay(replay_capacity_per_phase, data_seed + 3)
        self.rollout_phases = torch.zeros(
            self.storage.num_transitions_per_env, self.storage.num_envs, dtype=torch.long, device=self.device
        )
        self.teacher_mask = None
        self.phase_provider = None
        self.success_provider = None
        self.collecting = True
        self._reset_metrics()

    @staticmethod
    def teacher_probability(update):
        for end, probability in ((50, 1.0), (150, 0.5), (250, 0.25), (350, 0.1)):
            if update < end:
                return probability
        return 0.0

    def _reset_metrics(self):
        self.episode_counts = {
            f"{who}_{what}": 0 for who in ("student", "teacher") for what in ("episodes", "successes")
        }
        self.teacher_steps = 0
        self.total_steps = 0

    def _assign(self, count):
        return (torch.rand(count, generator=self.controller_rng) < self.teacher_probability(self.num_updates)).to(
            self.device
        )

    def act(self, obs):
        student_actions = self.student(obs, stochastic_output=self.collecting).detach().clamp(-1, 1)
        if not self.collecting:
            return student_actions
        if self.phase_provider is None:
            raise RuntimeError("Attach the box pre-action phase provider before collection")
        if self.teacher_mask is None:
            self.teacher_mask = self._assign(len(obs))
        labels = self.teacher(obs).detach().clamp(-1, 1)
        self.rollout_phases[self.storage.step].copy_(self.phase_provider())
        # The simulator can update its observation buffers in-place in step().
        self.transition.observations = obs.clone()
        self.transition.privileged_actions = labels
        self.transition.actions = torch.where(self.teacher_mask[:, None], labels, student_actions)
        self.teacher_steps += int(self.teacher_mask.sum())
        self.total_steps += len(obs)
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        done = dones.bool().flatten()
        if done.any():
            success = self.success_provider().bool().flatten()
            for who, mask in (("teacher", self.teacher_mask), ("student", ~self.teacher_mask)):
                ended = done & mask
                self.episode_counts[f"{who}_episodes"] += int(ended.sum())
                self.episode_counts[f"{who}_successes"] += int((ended & success).sum())
            self.teacher_mask[done] = self._assign(int(done.sum()))
        super().process_env_step(obs, rewards, dones, extras)

    def update(self):
        fresh = self.storage.observations.flatten(0, 1)
        labels = self.storage.privileged_actions.flatten(0, 1)
        phases = self.rollout_phases.flatten()
        losses, fresh_losses, replay_losses = [], [], []
        replay_counts = torch.zeros(4, dtype=torch.long)
        for _ in range(self.num_learning_epochs * self.storage.num_transitions_per_env):
            replay = self.replay.sample(64, self.device)
            fresh_count = 64 if replay is not None else 128
            indices = torch.randint(len(fresh), (fresh_count,), generator=self.fresh_rng).to(self.device)
            batch = TensorDict({key: fresh[key][indices] for key in STUDENT_KEYS}, [fresh_count], device=self.device)
            target = labels[indices]
            if replay is not None:
                replay_obs, replay_labels, replay_phases = replay
                batch = torch.cat((batch, replay_obs), dim=0)
                target = torch.cat((target, replay_labels))
                replay_counts += replay_phases.bincount(minlength=4)
            with torch.amp.autocast(
                device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
            ):
                prediction = self.student(batch)
                loss = self.loss_fn(prediction, target)
            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
            self.optimizer.step()
            losses.append(loss.item())
            fresh_losses.append(self.loss_fn(prediction[:fresh_count].detach(), target[:fresh_count]).item())
            if replay is not None:
                replay_losses.append(self.loss_fn(prediction[fresh_count:].detach(), target[fresh_count:]).item())
        # Neither replay nor teacher states contribute additional normalizer updates.
        self.student.update_normalization(fresh)
        self.replay.add(fresh, labels, phases)
        metrics = {
            "behavior": sum(losses) / len(losses),
            "fresh_behavior": sum(fresh_losses) / len(fresh_losses),
            "replay_behavior": sum(replay_losses) / len(replay_losses) if replay_losses else 0.0,
            "replay_samples": int(replay_counts.sum()),
            "relative_update": self.num_updates,
            "teacher_probability": self.teacher_probability(self.num_updates),
            "teacher_control_fraction": self.teacher_steps / max(1, self.total_steps),
            **self.episode_counts,
        }
        for who in ("teacher", "student"):
            count = self.episode_counts[f"{who}_episodes"]
            metrics[f"{who}_success_rate"] = self.episode_counts[f"{who}_successes"] / count if count else 0.0
        for phase, name in enumerate(PHASE_NAMES):
            metrics[f"fresh_phase_{name}"] = int((phases == phase).sum())
            metrics[f"replay_phase_{name}"] = int(replay_counts[phase])
            metrics[f"replay_occupancy_{name}"] = self.replay.sizes[phase]
        self.num_updates += 1
        self.storage.clear()
        self._reset_metrics()
        return metrics

    def train_mode(self):
        super().train_mode()
        self.collecting = True

    def eval_mode(self):
        super().eval_mode()
        self.collecting = False

    def save(self):
        saved = super().save()
        saved["box_data_first"] = {
            "version": 1,
            "num_updates": self.num_updates,
            "controller_rng": self.controller_rng.get_state().clone(),
            "fresh_rng": self.fresh_rng.get_state().clone(),
            "replay": self.replay.state_dict(),
        }
        return saved

    def load(self, loaded_dict, load_cfg, strict):
        result = super().load(loaded_dict, load_cfg, strict)
        state = loaded_dict.get("box_data_first")
        # A teacher-only import must never inherit an unrelated student's replay.
        restore = state is not None and (load_cfg is None or load_cfg.get("student", False))
        if restore:
            if state["version"] != 1:
                raise ValueError("Unsupported data-first checkpoint version")
            self.num_updates = state["num_updates"]
            self.replay.load_state_dict(state["replay"])
            self.controller_rng.set_state(state["controller_rng"].cpu())
            self.fresh_rng.set_state(state["fresh_rng"].cpu())
        else:
            self.num_updates = 0
            self.replay = PhaseReplay(self.replay.capacity, self.data_seed + 3)
            self.controller_rng.manual_seed(self.data_seed + 1)
            self.fresh_rng.manual_seed(self.data_seed + 2)
        self.teacher_mask = None
        self.storage.clear()
        self._reset_metrics()
        return result

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        from .mdp import progress

        alg = Distillation.construct_algorithm(obs, env, cfg, device)
        box = env.unwrapped
        alg.phase_provider = lambda: phase_ids(progress(box))
        alg.success_provider = lambda: box._box_terminal_progress[:, 3]
        return alg
