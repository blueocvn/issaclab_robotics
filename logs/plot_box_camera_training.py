import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

series=json.loads(Path('logs/box_camera_training_series.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,constrained_layout=True)
for ax,(tag,title,ylabel,factor) in zip(axes,[
    ('Loss/behavior','Student imitation loss','Action MSE',1),
    ('Episode_Termination/success','Training episode-success estimate','Percent',100),
    ('Train/mean_episode_length','Training episodes shorten as the student learns placement','Control steps (30 Hz)',1),
]):
    points=series[tag]
    x=np.array([p['iteration'] for p in points])
    y=np.array([p['value'] for p in points])*factor
    ax.plot(x,y,color='#287b8e',alpha=.2,linewidth=1)
    ax.plot(x[24:],np.convolve(y,np.ones(25)/25,mode='valid'),color='#006d77',linewidth=2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.2)
axes[1].set_ylim(-2,102)
axes[-1].set_xlabel('Camera-distillation update')
fig.suptitle('SO-101 box wrist-camera student | 1,600 distillation updates\nRaw data (light) and trailing 25-update means (solid)',fontsize=14)
fig.savefig('logs/box_camera_training_curves.png',dpi=160)
fig.savefig('logs/box_camera_training_curves.svg')
print('Saved camera training curves')
