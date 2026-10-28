"""Verify completed500-iteration native PPO continuation with60s horizon."""
from pathlib import Path
import hashlib
import json
import re
import torch

p=Path(__file__).resolve().parent
source=Path('logs/rsl_rl/so101_box_camera_ppo/2026-10-28_16-56-07_camera_ppo_2200_plus1000/model_1048.pt')
run=Path('logs/rsl_rl/so101_box_camera_ppo/20261028_60s_trial_camera_ppo_60s_plus500')
endpoint=run/'model_1547.pt'
a=torch.load(source,map_location='cpu',weights_only=False)
b=torch.load(endpoint,map_location='cpu',weights_only=False)
assert hashlib.sha256(source.read_bytes()).hexdigest()=='32952d1e6783603d81cf9b72b4c13ecbb4f2c329577d3bfa5151a7bf7990b0d2'
assert b['iter']==1547
steps=sorted({int(v['step']) for v in b['optimizer_state_dict']['state'].values() if 'step' in v})
assert steps==[24800]
assert b['optimizer_state_dict']['param_groups'][0]['lr']==1e-5
counts={}
for name in ('actor_state_dict','critic_state_dict'):
    assert a[name].keys()==b[name].keys()
    assert all(torch.isfinite(v).all() for v in b[name].values())
    previous=int(a[name]['obs_normalizer.count']); current=int(b[name]['obs_normalizer.count'])
    assert current==previous+500*128*64
    counts[name]={'before':previous,'after':current}
assert re.search(r'^episode_length_s: 60\.0\s*$',(run/'params/env.yaml').read_text(),re.M)
assert re.search(r'^max_iterations: 500\s*$',(run/'params/agent.yaml').read_text(),re.M)
log=(p/'train.log').read_text()
labels=[int(v) for v in re.findall(r'Learning iteration (\d+)/',log)]
assert labels==list(range(1048,1548))
report=dict(run=str(run),source=str(source),endpoint=str(endpoint),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),endpoint_sha256=hashlib.sha256(endpoint.read_bytes()).hexdigest(),additional_iterations=500,iteration_labels=[1048,1547],optimizer_steps=steps,normalizer_counts=counts,episode_length_s=60.0,episode_control_steps=1800,finite_actor_critic=True,learning_rate=1e-5)
assert not (p/'verification.json').exists()
(p/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
