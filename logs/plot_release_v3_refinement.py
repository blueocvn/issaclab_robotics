import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

original=json.loads(Path('logs/box_release_v3_training_series.json').read_text())
refined=json.loads(Path('logs/box_release_v3_refine_training_series.json').read_text())
plt.rcParams.update({'font.size':10, 'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,constrained_layout=True)
for ax,(tag,title,ylabel,factor) in zip(axes,[
    ('Episode_Termination/success','Training episode-success estimate','Percent',100),
    ('Train/mean_reward','Training mean reward','Reward',1),
    ('Train/mean_episode_length','Episodes shorten as successful placement becomes faster','Control steps (30 Hz)',1),
]):
    for label,series,color in [('Release-v3: 400 updates',original,'#718096'),('PPO refinement: 200 updates',refined,'#087f5b')]:
        points=series[tag]
        x=np.array([p['iteration'] for p in points])
        y=np.array([p['value'] for p in points])*factor
        ax.plot(x,y,color=color,alpha=.2,linewidth=1)
        ax.plot(x[9:],np.convolve(y,np.ones(10)/10,mode='valid'),color=color,label=label,linewidth=2)
    ax.axvline(1997,color='#be6b28',linestyle='--',linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.2)
axes[0].set_ylim(-2,102)
axes[0].legend(loc='upper left')
axes[-1].set_xlabel('Global PPO update; dashed line marks refinement restart')
fig.suptitle('SO-101 release-v3 and PPO refinement\nRaw iteration data (light) and trailing 10-update means (solid)',fontsize=14)
fig.savefig('logs/box_release_v3_refinement_curves.png',dpi=160)
fig.savefig('logs/box_release_v3_refinement_curves.svg')
print('Saved refinement comparison curves')
