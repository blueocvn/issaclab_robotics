import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

series = json.loads(Path('logs/box_release_v3_training_series.json').read_text())
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(12, 7.3), constrained_layout=True)

def curve(ax, tag, color, label, factor=1):
    points=series[tag]
    x=np.array([p['iteration'] for p in points])
    y=np.array([p['value'] for p in points])*factor
    ax.plot(x, y, color=color, alpha=.2, linewidth=1)
    smooth=np.convolve(y, np.ones(10)/10, mode='valid')
    ax.plot(x[9:], smooth, color=color, linewidth=2, label=label)
    ax.grid(alpha=.2)
    ax.set_xlabel('Global PPO update')

curve(axes[0,0], 'Train/mean_reward', '#006d77', 'Mean reward')
axes[0,0].set_title('Reward increased late in the run')
axes[0,0].set_ylabel('Training mean reward')
curve(axes[0,1], 'Episode_Termination/success', '#337844', 'Episode-success estimate', 100)
axes[0,1].axhline(90, color='#a33d3d', linestyle='--', linewidth=1, label='90% acceptance target')
axes[0,1].set_title('Training success estimate; audit decides acceptance')
axes[0,1].set_ylabel('Percent')
axes[0,1].set_ylim(-2,102)
axes[0,1].legend(fontsize=8)
curve(axes[1,0], 'Train/mean_episode_length', '#7356a1', 'Episode length')
axes[1,0].set_title('Episodes shorten as placement is learned')
axes[1,0].set_ylabel('Control steps (30 Hz)')
curve(axes[1,1], 'Metrics/on_target_rate', '#61708a', 'On target', 100)
curve(axes[1,1], 'Metrics/released_rate', '#d07423', 'Released', 100)
curve(axes[1,1], 'Metrics/withdrawn_rate', '#a23f75', 'Withdrawn', 100)
axes[1,1].set_title('Live stage occupancy; successful episodes reset')
axes[1,1].set_ylabel('Percent of active batch')
axes[1,1].legend(fontsize=8)
fig.suptitle('SO-101 box release-v3 | 400 updates: 1598–1997\nLight lines: raw iteration data; solid lines: trailing 10-update mean', fontsize=14)
fig.savefig('logs/box_release_v3_training_curves.png', dpi=160)
fig.savefig('logs/box_release_v3_training_curves.svg')
print('Saved training curves PNG and SVG')
