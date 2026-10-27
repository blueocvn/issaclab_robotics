"""Read-only analysis of completed overnight training and saved exact audits."""
import json
import math
from collections import Counter
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(__file__).resolve().parent
run = root / 'overnight_20261027_ppo_camera'
events = [json.loads(line) for line in (run/'journal.jsonl').read_text().splitlines()]
training = [r for r in events if r['event'] == 'training_completed']
audits = [r for r in events if r['event'] == 'audit_completed']
result = dict(training_blocks=len(training), additional_ppo_updates=len(training)*200,
              accepted_stages=[r['accepted_stage'] for r in events if r['event'] == 'stage_accepted'],
              camera_started=any(r.get('phase', '').startswith('camera') for r in events), stages=[], failures={})
f_starts = None
failed_sets = []
for row in audits:
    directory = Path(row['audit_directory'])
    # Stored Linux paths map to the current repository's logs on either mount.
    directory = root / directory.relative_to('/mnt/d/work/robotics/IsaacLabTutorial/logs')
    data = json.loads((directory/'group43_batch0.json').read_text())
    rows = data['episodes']
    assert len(rows) == 1024 and {r['env_id'] for r in rows} == set(range(1024))
    for key, value in data['summary'].items():
        if key.endswith('_rate'):
            assert abs(sum(r[key[:-5]] for r in rows)/1024-value) < 1e-12
    name = directory.name.removesuffix('_stage_try0')
    summary = row['last_audit']
    spatial = [(summary['spatial_successes'][iy][ix]/count, ix, iy, count)
               for iy, line in enumerate(summary['spatial_counts']) for ix, count in enumerate(line) if count]
    yaw = [(win/count, i, count) for i, (win, count) in
           enumerate(zip(summary['yaw_successes'], summary['yaw_counts'], strict=True)) if count]
    result['stages'].append(dict(name=name, successes=sum(r['success'] for r in rows),
                                 min_spatial=min(spatial), min_yaw=min(yaw)))
    if '_F_' in name:
        starts = [r['initial_box_pose'] for r in sorted(rows, key=lambda r:r['env_id'])]
        if f_starts is None:
            f_starts = starts
        else:
            assert all(abs(x-y) <= 1e-7 for a,b in zip(starts, f_starts, strict=True)
                       for x,y in zip(a,b,strict=True))
        failed = [r for r in rows if not r['success']]
        failed_sets.append({r['env_id'] for r in failed})
        result['failures'][name] = dict(
            counts=dict(never_grasped=sum(not r['grasped'] for r in failed),
                        grasped_without_lift=sum(r['grasped'] and not r['lifted'] for r in failed),
                        lifted_without_success=sum(r['lifted'] for r in failed)),
            ids=[r['env_id'] for r in failed],
            starts=[dict(env_id=r['env_id'], xy=r['initial_box_pose'][:2],
                         yaw_deg=math.degrees(2*math.atan2(r['initial_box_pose'][5],r['initial_box_pose'][6])),
                         grasped=r['grasped'], lifted=r['lifted'], on_target=r.get('on_target'),
                         upright=r.get('upright'), released=r.get('released'), withdrawn=r.get('withdrawn')) for r in failed])
result['all_F_initial_poses_match'] = True
result['persistent_F_failures'] = sorted(set.intersection(*failed_sets))
counts = Counter(i for failures in failed_sets for i in failures)
result['most_frequent_F_failures'] = counts.most_common(12)
result['training'] = []
for row in training:
    directory = root / Path(row['training_run']).relative_to('/mnt/d/work/robotics/IsaacLabTutorial/logs')
    ea = EventAccumulator(str(directory), size_guidance={'scalars':0}).Reload()
    tags = ea.Tags()['scalars']
    values = {tag:[v.value for v in ea.Scalars(tag)] for tag in tags}
    assert values and all(math.isfinite(v) for series in values.values() for v in series)
    lr_tag = next((tag for tag in tags if 'learning_rate' in tag.lower()), None)
    result['training'].append(dict(run=directory.name,
        all_scalars_finite=True,
        learning_rate=dict(first=values[lr_tag][0], last=values[lr_tag][-1],
                           minimum=min(values[lr_tag]), maximum=max(values[lr_tag])) if lr_tag else None))
(root/'overnight_workspace_analysis.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('training','failures')}, indent=2))
print('F6_TRAINING=' + json.dumps(result['training'][-3]))
print('F6_FAILURES=' + json.dumps(result['failures']['ppo_F_block6']['counts']))
