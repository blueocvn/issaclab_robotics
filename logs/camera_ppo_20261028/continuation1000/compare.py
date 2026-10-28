"""Paired selection screens of three saved PPO continuation checkpoints."""
import hashlib
import json
from pathlib import Path
from isaaclab_tutorial.tasks.place_box.camera_comparison import load_population, paired_summary

p=Path(__file__).resolve().parent
parent=p.parent
verification=json.loads((p/'verification.json').read_text())
report={'method':'1000 additional Camera PPO iterations; three selected checkpoint screens','evaluated_labels':[100,700,1048],'baseline':'fresh native2200 actor, before PPO','baseline_sha256':None,'candidates':{},'limitations':['Three of11 saves screened; no global-best claim.','256-start selection screens are not deployment acceptance.','Training metrics nominate candidates only; independent confirmation required for adoption.']}
for label in report['evaluated_labels']:
    candidate={}
    for name,pop in [('nominal','nominal'),('wide','stage')]:
        a,ap,ah=load_population(parent/f'initial_{name}',pop,256)
        b,bp,bh=load_population(p/f'c{label}_{name}',pop,256)
        assert ap==bp,'Protocol mismatch'
        assert bh==verification['checkpoints'][str(label)]['sha256']
        assert hashlib.sha256(Path(verification['checkpoints'][str(label)]['path']).read_bytes()).hexdigest()==bh
        if report['baseline_sha256'] is None: report['baseline_sha256']=ah
        assert ah==report['baseline_sha256']
        candidate[name]=paired_summary(a,b)
        candidate['checkpoint_sha256']=bh
    candidate['nominal_eligible']=candidate['nominal']['gain_pp']>=-3
    candidate['qualifies']=candidate['nominal_eligible'] and candidate['wide']['gain_pp']>=10
    report['candidates'][str(label)]=candidate
eligible=[label for label,c in report['candidates'].items() if c['nominal_eligible']]
report['best_nominal_eligible_screened']=max(eligible,key=lambda label:report['candidates'][label]['wide']['candidate_successes']) if eligible else None
qualified=[label for label,c in report['candidates'].items() if c['qualifies']]
report['confirmation_candidate']=max(qualified,key=lambda label:report['candidates'][label]['wide']['candidate_successes']) if qualified else None
report['decision']='Confirm selected candidate on independent1024-start cohorts' if qualified else 'No screened checkpoint qualifies; retain2200. No further training started.'
assert not (p/'comparison.json').exists()
(p/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
