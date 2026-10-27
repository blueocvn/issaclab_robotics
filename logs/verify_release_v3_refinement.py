"""Cross-check refinement audits and compare the previously failed start."""
import json
from pathlib import Path

old=json.loads(Path('logs/box_release_v3_1997_evaluation.json').read_text())
old_rows={r['env_id']:r for r in old['episodes']}
audits={}
all_outcomes=[]
for seed in (42,43,44):
    file=Path(f'logs/box_release_v3_refine_2196_seed{seed}_evaluation.json')
    audit=json.loads(file.read_text())
    rows=audit['episodes']
    assert len(rows)==1024 and {r['env_id'] for r in rows}==set(range(1024))
    for field in ('grasped','lifted','placed','success','on_target','released','withdrawn','upright','timed_out','dropped','out_of_bounds','unstable_robot'):
        assert audit['summary'][field+'_rate']==sum(bool(r[field]) for r in rows)/1024
    by_id={r['env_id']:r for r in rows}
    matching_starts=sum(max(abs(a-b) for a,b in zip(old_rows[i]['initial_box_pose'],row['initial_box_pose']))<1e-7
                        for i,row in by_id.items())
    if seed==42:
        assert matching_starts==1024
        previously_failed=by_id[527]
    else:
        assert matching_starts<512, 'Extra seed did not substantially change reset samples'
    audits[str(seed)]=dict(summary=audit['summary'], matches_original_starts=matching_starts,
                          failed_ids=[r['env_id'] for r in rows if not r['success']])
    all_outcomes.extend(rows)
training=json.loads(Path('logs/box_release_v3_refine_training_summary.json').read_text())
assert all(v['finite'] for v in training['scalars'].values())
assert training['scalars']['Episode_Termination/success']['count']==200
report=dict(checkpoint='logs/rsl_rl/so101_box_state/2026-10-26_14-59-07_so101-box-release-v3-refine/model_2196.pt',
            audits=audits, total_episodes=len(all_outcomes), successes=sum(r['success'] for r in all_outcomes),
            old_case_527_success=old_rows[527]['success'], refined_case_527=previously_failed,
            finite_training_scalars=True)
Path('logs/box_release_v3_refinement_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,sort_keys=True))
