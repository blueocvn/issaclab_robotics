"""Verify collection integrity and generate review artifacts without training."""
import hashlib
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
bundles={name:torch.load(P/f'{name}_collection.pt',map_location='cpu',weights_only=False) for name in ('baseline','teacher')}
baseline,teacher=[bundles[n] for n in ('baseline','teacher')]
assert baseline['provenance']['checkpoint_sha256']==teacher['provenance']['checkpoint_sha256']
assert torch.allclose(torch.tensor(baseline['provenance']['initial_poses']),torch.tensor(teacher['provenance']['initial_poses']),atol=2e-6,rtol=0)
summary={'matched_initial_poses':True,'optimizer_updates':0,'collections':{},'label_sources':baseline['label_sources']}
for name,data in bundles.items():
    assert len(data['episodes'])==48
    assert len({e['env_id'] for e in data['episodes']})==48
    snapshots=0
    for e in data['episodes']:
        n=len(e['steps']); snapshots+=n
        assert n>0 and e['steps'][0]==0
        assert e['baseline_actions'].shape==e['teacher_actions'].shape==(n,6)
        assert e['wrist_rgb'].shape[0]==e['proprioception'].shape[0]==len(e['phase'])==n
        assert all(torch.isfinite(e[k]).all() for k in ('wrist_rgb','proprioception','baseline_actions','teacher_actions'))
        assert all(e[k].abs().max()<=1 for k in ('baseline_actions','teacher_actions'))
        assert e['phase'].min()>=0 and e['phase'].max()<=3
    summary['collections'][name]=dict(snapshots=snapshots,successful_nominal=sum(e['population']=='nominal' and e['terminal_flags'][3] for e in data['episodes']),successful_wide=sum(e['population']=='wide' and e['terminal_flags'][3] for e in data['episodes']),strata=data['provenance']['counts'],sha256=hashlib.sha256((P/f'{name}_collection.pt').read_bytes()).hexdigest())
fig,axes=plt.subplots(6,8,figsize=(16,12),constrained_layout=True)
for ax,e in zip(axes.flat,baseline['episodes'],strict=True):
    frame=e['wrist_rgb'][0].float()
    if frame.ndim==4: frame=frame[-1]
    if frame.shape[0] in (3,6): frame=frame[-3:].permute(1,2,0)
    assert frame.ndim==3 and frame.shape[-1]==3,frame.shape
    ax.imshow(frame.clamp(0,1).numpy(),interpolation='nearest')
    ax.set_title(f"{e['env_id']}: {e['stratum']}\n2200 {'success' if e['terminal_flags'][3] else 'failure'}",fontsize=7)
    ax.axis('off')
fig.savefig(P/'initial_policy_views.png',dpi=140)
(P/'collection_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
