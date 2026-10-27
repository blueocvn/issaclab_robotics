"""Verify the bounded Stage A run and save compact runtime evidence."""
import json
import math
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(__file__).resolve().parent
source = root/'rsl_rl/so101_box_state/2026-10-26_14-59-07_so101-box-release-v3-refine/model_2196.pt'
run = root/'rsl_rl/so101_box_state/2026-10-26_16-55-57_so101-box-workspace-A-smoke-fixed'
checkpoint = run/'model_2197.pt'
old = torch.load(source, map_location='cpu', weights_only=False)
new = torch.load(checkpoint, map_location='cpu', weights_only=False)
def optimizer_steps(data):
    return [float(v['step']) for v in data['optimizer_state_dict']['state'].values() if 'step' in v]
steps_old, steps_new = optimizer_steps(old), optimizer_steps(new)
assert max(steps_new) == 80 and min(steps_new) == 80
assert max(steps_old) > max(steps_new)
events = EventAccumulator(str(run), size_guidance={'scalars': 0}).Reload()
scalars = {tag: [v.value for v in events.Scalars(tag)] for tag in events.Tags()['scalars']}
assert scalars and all(math.isfinite(x) for values in scalars.values() for x in values)
assert all(len(values) == 2 for tag, values in scalars.items() if tag.startswith('Loss/'))
probe = json.loads((root/'workspace_A_smoke_reset.json').read_text())
pose = torch.tensor(probe['poses'])
sources = torch.tensor(probe['sources'])
xy = pose[:, :2] - torch.tensor([.17, -.1])
assert len(pose) == 512 and set(sources.tolist()) == {0, 1}
assert xy[sources == 0].abs().max() <= .0101
assert xy[sources == 1].abs().max() <= .0201
assert torch.allclose(pose[:, 3:].norm(dim=-1), torch.ones(512), atol=1e-6)
assert (2*torch.atan2(pose[:, 5], pose[:, 6])).abs().max() <= .10001
assert probe['parameters']['nominal_fraction'] == .2
result = dict(run=str(run), checkpoint=str(checkpoint), loaded_teacher=str(source),
              optimizer_step_before=max(steps_old), optimizer_step_after=max(steps_new),
              all_training_scalars_finite=True, loss_sample_counts={k:len(v) for k,v in scalars.items() if k.startswith('Loss/')},
              reset_count=len(pose), nominal_count=int((sources == 0).sum()), current_stage_count=int((sources == 1).sum()),
              nominal_max_xy_offset_m=float(xy[sources == 0].abs().max()),
              stage_max_xy_offset_m=float(xy[sources == 1].abs().max()),
              intended_mixture_and_ranges_verified=True,
              note='Two-update integration check; no policy competency or workspace acceptance claim.')
(root/'workspace_A_smoke_verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
