"""Matched60-second initial/endpoint physical-success selection screens."""
from pathlib import Path
import hashlib
import json
from isaaclab_tutorial.tasks.place_box.camera_comparison import load_population, paired_summary

p=Path(__file__).resolve().parent
verification=json.loads((p/'verification.json').read_text())
report={'method':'500 additional Camera PPO iterations with60-second episodes','populations':{},'checkpoint_sha256':{},'same_episode_horizon_seconds':60.0,'limitations':['256-start selection screens, not deployment acceptance.','30-second historical scores cannot be treated as a matched baseline for60-second success.']}
for name,pop in [('nominal','nominal'),('wide','stage')]:
    a,ap,ah=load_population(p/f'before_{name}',pop,256)
    b,bp,bh=load_population(p/f'after_{name}',pop,256)
    assert ap==bp
    assert all('env.episode_length_s=60.0' in command for command in ap)
    assert ah==verification['source_sha256'] and bh==verification['endpoint_sha256']
    report['populations'][name]=paired_summary(a,b)
    report['checkpoint_sha256'][name]={'source':ah,'endpoint':bh}
assert report['checkpoint_sha256']['nominal']==report['checkpoint_sha256']['wide']
assert hashlib.sha256(Path(verification['source']).read_bytes()).hexdigest()==verification['source_sha256']
assert hashlib.sha256(Path(verification['endpoint']).read_bytes()).hexdigest()==verification['endpoint_sha256']
report['qualifies_on_same_horizon']=report['populations']['nominal']['gain_pp']>=-3 and report['populations']['wide']['gain_pp']>=10
assert not (p/'comparison.json').exists()
(p/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
