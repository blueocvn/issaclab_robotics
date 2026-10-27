import json
from pathlib import Path

files={'latest_first':'box_camera_1599_seed42_evaluation.json',
       'latest_repeat':'box_camera_1599_seed42_repeat_evaluation.json',
       'previous':'box_camera_1500_seed42_evaluation.json'}
audits={}
reference=None
for label,name in files.items():
    audit=json.loads((Path('logs')/name).read_text())
    rows=audit['episodes']
    by_id={r['env_id']:r for r in rows}
    assert len(rows)==len(by_id)==1024 and set(by_id)==set(range(1024))
    for field in ('grasped','lifted','placed','success','released','withdrawn','on_target','upright','timed_out','dropped','out_of_bounds','unstable_robot'):
        assert audit['summary'][field+'_rate']==sum(r[field] for r in rows)/1024
    if reference is None:
        reference=by_id
    assert all(max(abs(a-b) for a,b in zip(reference[i]['initial_box_pose'],row['initial_box_pose']))<1e-7
               for i,row in by_id.items())
    audits[label]=dict(summary=audit['summary'],successes=sum(r['success'] for r in rows),
                       failed_ids=[r['env_id'] for r in rows if not r['success']])
training=json.loads(Path('logs/box_camera_training_summary.json').read_text())
assert all(s['finite'] for s in training['scalars'].values())
assert training['scalars']['Loss/behavior']['count']==1600
trace=json.loads(Path('logs/box_camera_1599_failed_episodes_trace.json').read_text())
assert trace['initial_poses_match']
report=dict(audits=audits,all_audit_initial_poses_match=True,finite_training_scalars=True,
            instrumented_replay_terminal_successes={k:v['success'] for k,v in trace['terminal'].items()},
            latest_pooled_successes=audits['latest_first']['successes']+audits['latest_repeat']['successes'],
            latest_pooled_episodes=2048)
Path('logs/box_camera_validation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,sort_keys=True))
