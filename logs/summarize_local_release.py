import json
from pathlib import Path

path = Path('logs/local_pc_scripted_release.json')
data = json.loads(path.read_text())
outcomes = {}
for row in data['trace']:
    for idx, done in enumerate(row['done']):
        if done and row['episode_ids'][idx] == 0 and idx not in outcomes:
            outcomes[idx] = dict(env_id=idx, step=row['step'], success=row['success'][idx],
                                 grasped=row['grasped'][idx], lifted=row['lifted'][idx],
                                 bottom_height_m=row['bottom_height'][idx])
summary = dict(episodes=len(outcomes), successes=sum(o['success'] for o in outcomes.values()),
               scripted_release_at=data['scripted_release_at'], episodes_detail=list(outcomes.values()))
Path('logs/local_pc_scripted_release_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
assert len(outcomes) == 8 and all(o['success'] and o['grasped'] and o['lifted'] for o in outcomes.values())
