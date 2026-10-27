"""Check that camera smoke training used the exact requested frozen teacher."""
import hashlib
import json
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(__file__).resolve().parents[1]
teacher_path = root / "logs/rsl_rl/so101_box_state/2026-10-27_03-19-50_so101-box-workspace-F-refine/model_5381.pt"
runs = sorted((root / "logs/rsl_rl/so101_box_camera_distillation").glob("*_so101-box-workspace-camera-5381-smoke"))
run = runs[-1]
models = sorted(run.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
assert len(models) >= 5, "Five smoke iterations must produce five snapshots"
first = torch.load(models[0], map_location="cpu", weights_only=False)
last = torch.load(models[-1], map_location="cpu", weights_only=False)
teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)["actor_state_dict"]
embedded = last["teacher_state_dict"]
assert teacher.keys() == embedded.keys()
assert all(torch.equal(value, embedded[key]) for key, value in teacher.items()), "Wrong or modified teacher"
assert all(torch.equal(first["teacher_state_dict"][key], value) for key, value in embedded.items())
assert all(torch.isfinite(value).all() for value in last["student_state_dict"].values() if value.is_floating_point())
changed = [key for key, value in last["student_state_dict"].items()
           if not torch.equal(value, first["student_state_dict"][key])]
assert changed, "Student parameters must update"
events = EventAccumulator(str(run)).Reload()
loss_tags = [tag for tag in events.Tags()["scalars"] if "loss" in tag.lower() or "behavior" in tag.lower()]
assert loss_tags, "Distillation loss must be logged"
losses = {tag: [dict(step=point.step, value=point.value) for point in events.Scalars(tag)] for tag in loss_tags}
assert all(torch.isfinite(torch.tensor([point["value"] for point in points])).all() for points in losses.values())
result = dict(run=str(run.relative_to(root)), checkpoint=str(models[-1].relative_to(root)),
              teacher_sha256=hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
              embedded_teacher_matches_5381=True, teacher_unchanged=True,
              student_tensors_changed=len(changed), student_finite=True, loss_series=losses)
(root / "logs/camera_5381_smoke_verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
