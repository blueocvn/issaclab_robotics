"""Plot saved exact-episode results; no training or simulation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = json.loads((root/'overnight_workspace_analysis.json').read_text())
rows = [r for r in data['stages'] if r['name'].startswith('ppo_F_')]
x = list(range(1, len(rows)+1))
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
axes[0].plot(x, [r['successes']/1024*100 for r in rows], 'o-', color='#2166ac')
axes[0].axhline(99, color='#b2182b', linestyle='--', label='99% promotion gate')
axes[0].set_ylabel('Full success (%)')
axes[0].set_ylim(80, 100.5)
axes[0].legend(loc='lower right')
axes[1].plot(x, [r['min_spatial'][0]*100 for r in rows], 'o-', label='Worst spatial bin')
axes[1].plot(x, [r['min_yaw'][0]*100 for r in rows], 's-', label='Worst yaw bin')
axes[1].axhline(95, color='#b2182b', linestyle='--', label='95% bin gate')
axes[1].set_ylabel('Full success in worst bin (%)')
axes[1].legend(loc='lower right')
for ax in axes:
    ax.set_xlabel('Stage F training block (200 PPO updates each)')
    ax.set_xticks(x)
    ax.grid(alpha=.2)
fig.suptitle('Overnight Stage F: ±10 cm in X/Y, full upright rotation\n1,024 matched starts per checkpoint, selection seed43')
fig.savefig(root/'overnight_workspace_results.png', dpi=170)
fig.savefig(root/'overnight_workspace_results.svg')
