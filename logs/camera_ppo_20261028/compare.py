"""Compare native Camera PPO initial/endpoint and check historical2200 parity."""
import hashlib
import json
from pathlib import Path
import torch
from isaaclab_tutorial.tasks.place_box.camera_comparison import load_population, paired_summary

p=Path(__file__).resolve().parent
source=Path('logs/rsl_rl/so101_box_camera_distillation/2026-10-27_16-26-29_so101-box-workspace-camera-1798-lr1e-4-seed43/model_2200.pt')
historical=Path('logs/camera_800_gate_20261027_170933/comparison_cli')
original=torch.load(source,map_location='cpu',weights_only=False)
initial=torch.load(p/'trial_v2/initial.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(v,original['student_state_dict'][k]) for k,v in initial['actor_state_dict'].items())
assert not initial['optimizer_state_dict']['state']
report=dict(method='Camera PPO reward fine-tuning from2200',iterations=50,optimizer_steps=800,populations={},historical_initial_comparison={},checkpoint_sha256={},source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),initial_actor_all_tensors_equal=True,initial_optimizer_fresh=True)
for name,pop,old in [('nominal','nominal','2200_nominal'),('wide','stage','2200_stage')]:
    a,ap,ah=load_population(p/f'initial_{name}',pop,256)
    b,bp,bh=load_population(p/f'endpoint_{name}',pop,256)
    h,hp,hh=load_population(historical/old,pop,256)
    assert ap==bp,'PPO initial/endpoint protocols differ'
    normalized=[]
    for command in ap:
        command=command.copy()
        position=command.index('--agent')
        assert command[position+1]=='rsl_rl_camera_ppo_cfg_entry_point'
        del command[position:position+2]
        normalized.append(command)
    assert normalized==hp,'Historical protocol differs beyond explicit native PPO agent'
    assert hh==report['source_sha256']
    report['populations'][name]=paired_summary(a,b)
    report['historical_initial_comparison'][name]=paired_summary(h,a)
    report['checkpoint_sha256'][name]=dict(initial=ah,endpoint=bh)
assert report['checkpoint_sha256']['nominal']==report['checkpoint_sha256']['wide']
assert report['checkpoint_sha256']['nominal']['initial']==hashlib.sha256((p/'trial_v2/initial.pt').read_bytes()).hexdigest()
assert report['checkpoint_sha256']['nominal']['endpoint']==hashlib.sha256((p/'trial_v2/training/model_49.pt').read_bytes()).hexdigest()
report['eligible_for_confirmation']=report['populations']['nominal']['gain_pp']>=-3 and report['populations']['wide']['gain_pp']>=10 and report['populations']['nominal']['candidate_successes']>=222 and report['populations']['wide']['candidate_successes']>=106
report['decision']='Confirm once on independent1024 starts' if report['eligible_for_confirmation'] else 'Retain2200; bounded Camera PPO trial did not qualify. No automatic extension.'
assert not (p/'comparison.json').exists()
(p/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
