"""Compare final-clean candidate using existing protocol and paired-pose validators."""
import json
from pathlib import Path
import torch
from isaaclab_tutorial.tasks.place_box.camera_comparison import load_population, paired_summary

p=Path(__file__).resolve().parent
root=p.parent
old=json.loads((root/'training/verification_0100.json').read_text())
new=json.loads((p/'training/verification_0100.json').read_text())
assert new['optimizer_updates']==old['optimizer_updates']==100
assert new['optimizer_steps_before']==old['optimizer_steps_before']==[77184]
assert new['optimizer_steps_after']==old['optimizer_steps_after']==[77284]
assert new['teacher_unchanged'] and new['normalizer_unchanged'] and new['finite_gradients']
assert new['preservation_observation_corruption'] and not any(new['preservation_observation_corruption'].values())
old_improvement={k:v for k,v in old['dataset_sha256'].items() if '/expanded/' not in k}
new_improvement={k:v for k,v in new['dataset_sha256'].items() if '/final_clean/' not in k and not k.startswith('logs/camera_wide_focus_20261028/final_clean/')}
assert old_improvement==new_improvement and len(old_improvement)==2
old_poses=json.loads((root/'expanded/baseline_collection.json').read_text())['initial_poses']
new_poses=json.loads((p/'collection/baseline_collection.json').read_text())['initial_poses']
assert old_poses==new_poses
old_checkpoint=torch.load(root/'training/model_offline_0100.pt',map_location='cpu',weights_only=False)
new_checkpoint=torch.load(p/'training/model_offline_0100.pt',map_location='cpu',weights_only=False)
for k in ('lr','objective','normalizer_frozen','baseline_sha256'):
    assert old_checkpoint['offline_preservation'][k]==new_checkpoint['offline_preservation'][k]
assert old_checkpoint['optimizer_state_dict']['param_groups']==new_checkpoint['optimizer_state_dict']['param_groups']
report=dict(method='final-clean-preservation',optimizer_updates=100,single_experimental_change='Clean preservation observations at identical collection poses',unchanged_improvement_dataset_hashes=old_improvement,all_corruption_disabled=True,identical_preservation_poses=True,optimizer_settings_match_previous=True,populations={},checkpoint_sha256={})
b=Path('logs/camera_800_gate_20261027_170933/comparison_cli')
for key,pop,base,candidate in [('nominal','nominal',b/'2200_nominal',p/'screen_nominal'),('wide','stage',b/'2200_stage',p/'screen_wide')]:
    a,ap,ah=load_population(base,pop,256)
    c,cp,ch=load_population(candidate,pop,256)
    assert ap==cp
    report['populations'][key]=paired_summary(a,c)
    report['checkpoint_sha256'][key]={'baseline':ah,'candidate':ch}
assert report['checkpoint_sha256']['nominal']==report['checkpoint_sha256']['wide']
report['eligible_for_confirmation']=report['populations']['nominal']['gain_pp']>=-3 and report['populations']['wide']['gain_pp']>=10
report['decision']='Run independent1024-start confirmation once' if report['eligible_for_confirmation'] else 'Final attempt failed selection; retain2200 and stop. No continuation or further training attempt.'
output=p/'comparison.json'
assert not output.exists()
output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
