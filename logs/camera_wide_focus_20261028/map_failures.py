"""Descriptive audit analysis. Audit poses remain evaluation-only."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/'logs/camera_800_gate_20261027_170933'
projection_path=ROOT/'logs/camera_visibility_diagnostic_20261027_153417/home_workspace_projection.json'
h=np.array(json.loads(projection_path.read_text())['home_projection_homography'])

def summarize(rows):
    counts=dict(Counter(r['outcome'] for r in rows))
    return dict(n=len(rows),counts=counts,success_rate=counts.get('success',0)/len(rows) if rows else None)

result={'limitations':['256-start selection screens, not final acceptance.', 'Historical home-plane center projection; not actual rollout visibility or occlusion.', 'Bins are descriptive, small and confounded; no causal conclusion.', 'Audit poses stay evaluation-only; training collection must use fresh seeds.'], 'sources':{}}
all_rows={}
for pop,filename in [('wide','2200_stage.json'),('nominal','2200_nominal.json')]:
    path=SOURCE/filename
    data=json.loads(path.read_text())
    assert len(data['episodes'])==256 and len({e['env_id'] for e in data['episodes']})==256
    rows=[]
    for e in data['episodes']:
        p=e['initial_box_pose']
        assert .069<p[0]<.271 and -.201<p[1]<.001
        uv=h@np.array([p[0],p[1],1.])
        u,v=uv[:2]/uv[2]
        yaw=math.degrees(2*math.atan2(p[5],p[6]))
        status='success' if e['success'] else 'no_grasp' if not e['grasped'] else 'grasp_no_lift' if not e['lifted'] else 'lift_no_success'
        rows.append(dict(env_id=e['env_id'],initial_pose=p,x=p[0],y=p[1],yaw=yaw,yaw_bin=min(int(abs(yaw)//45),3),outcome=status,projected_center=[float(u),float(v)],home_center_inside=bool(0<=u<64 and 0<=v<48)))
    assert sum(r['outcome']=='success' for r in rows)==round(data['summary']['success_rate']*256)
    all_rows[pop]=rows
    result['sources'][pop]=dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reset_parameters=data['reset_parameters'])
    result[pop]=dict(overall=summarize(rows),by_home_projection={str(v):summarize([r for r in rows if r['home_center_inside']==v]) for v in (True,False)},by_absolute_yaw={f'{i*45}-{(i+1)*45}':summarize([r for r in rows if r['yaw_bin']==i]) for i in range(4)})
rows=all_rows['wide']
xe,ye=np.linspace(.07,.27,5),np.linspace(-.2,0,5)
cells=[]
rates=np.full((4,4),np.nan)
for iy in range(4):
    for ix in range(4):
        selected=[r for r in rows if xe[ix]<=r['x']<xe[ix+1] and ye[iy]<=r['y']<ye[iy+1]]
        stats=summarize(selected)
        cells.append(dict(x_range=xe[ix:ix+2].tolist(),y_range=ye[iy:iy+2].tolist(),**stats))
        if selected: rates[iy,ix]=stats['success_rate']*100
assert sum(c['n'] for c in cells)==256
result['wide']['xy_cells']=cells
result['projection_source']=dict(path=str(projection_path.relative_to(ROOT)),sha256=hashlib.sha256(projection_path.read_bytes()).hexdigest())
fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
for key,color in {'success':'#198754','no_grasp':'#d9534f','grasp_no_lift':'#e5a126','lift_no_success':'#7655a3'}.items():
    sel=[r for r in rows if r['outcome']==key]
    axes[0].scatter([r['x'] for r in sel],[r['y'] for r in sel],s=24,color=color,label=f'{key.replace("_"," ")} ({len(sel)})',alpha=.8)
xx,yy=np.meshgrid(np.linspace(.07,.27,101),np.linspace(-.2,0,101))
uv=h@np.stack([xx.ravel(),yy.ravel(),np.ones(xx.size)])
u,v=(uv[:2]/uv[2]).reshape(2,*xx.shape)
inside=(u>=0)&(u<64)&(v>=0)&(v<48)
axes[0].contour(xx,yy,inside.astype(float),levels=[.5],colors='black',linestyles='--')
axes[0].legend(fontsize=8,loc='upper left')
axes[0].set_title('Wide outcomes; dashed = estimated initial-view boundary')
im=axes[1].imshow(rates,origin='lower',extent=[.07,.27,-.2,0],vmin=0,vmax=100,cmap='RdYlGn',aspect='auto')
for c in cells:
    axes[1].text(np.mean(c['x_range']),np.mean(c['y_range']),f"{c['counts'].get('success',0)}/{c['n']}",ha='center',va='center')
axes[1].set_title('Successes / episodes in 5 cm cells')
fig.colorbar(im,ax=axes[1],label='Success (%)')
for ax in axes: ax.set(xlabel='Box local X (m)',ylabel='Box local Y (m)',xlim=(.07,.27),ylim=(-.2,0))
fig.savefig(OUT/'wide_failure_map.png',dpi=170)
(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'episodes.json').write_text(json.dumps(all_rows,indent=2)+'\n')
print(json.dumps({p:{k:v for k,v in result[p].items() if k!='xy_cells'} for p in ('wide','nominal')},indent=2))
