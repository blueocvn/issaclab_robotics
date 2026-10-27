"""Summarize frozen-policy workspace screening; no simulation modifications."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CASES = [
    ('teacher_pm5cm', 'Teacher: +/-5 cm, yaw +/-5.7 deg',
     'box_workspace_teacher_2196_pm5cm_seed42_evaluation.json', .05, .1),
    ('teacher_pm10cm_full_yaw', 'Teacher: +/-10 cm, full upright yaw',
     'box_workspace_teacher_2196_pm10cm_full_yaw_seed42_evaluation.json', .1, math.pi),
    ('camera_pm10cm_full_yaw', 'Camera: +/-10 cm, full upright yaw',
     'box_workspace_camera_1599_pm10cm_full_yaw_seed42_evaluation.json', .1, math.pi),
]
results = {}
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)
for ax, (key, title, filename, halfwidth, maxyaw) in zip(axes, CASES, strict=True):
    path = ROOT / filename
    if not path.exists():
        ax.set_title(title + '\nPending')
        ax.axis('off')
        continue
    data = json.loads(path.read_text())
    rows = data['episodes']
    assert len(rows) == data['summary']['episodes']
    assert len({r['env_id'] for r in rows}) == len(rows)
    for name, value in data['summary'].items():
        if name.endswith('_rate'):
            assert abs(sum(r[name[:-5]] for r in rows) / len(rows) - value) < 1e-12
    total = np.zeros((5, 5), dtype=int)
    passed = np.zeros_like(total)
    yaw_total = np.zeros(8, dtype=int)
    yaw_passed = np.zeros_like(yaw_total)
    offsets = []
    for row in rows:
        pose = row['initial_box_pose']
        dx, dy = pose[0] - .17, pose[1] + .1
        yaw = 2 * math.atan2(pose[5], pose[6])
        assert abs(dx) <= halfwidth + 1e-4 and abs(dy) <= halfwidth + 1e-4
        assert abs(yaw) <= maxyaw + 1e-5
        offsets.append((dx, dy, yaw))
        ix = min(4, max(0, int((dx + halfwidth) / (2 * halfwidth) * 5)))
        iy = min(4, max(0, int((dy + halfwidth) / (2 * halfwidth) * 5)))
        total[iy, ix] += 1
        passed[iy, ix] += int(row['success'])
        iz = min(7, max(0, int((yaw + maxyaw) / (2 * maxyaw) * 8)))
        yaw_total[iz] += 1
        yaw_passed[iz] += int(row['success'])
    assert int(total.sum()) == len(rows) == int(yaw_total.sum())
    a = np.asarray(offsets)
    result = dict(
        source=filename, summary=data['summary'],
        successes=sum(r['success'] for r in rows),
        failure_stages=dict(
            never_grasped=sum(not r['grasped'] for r in rows),
            grasped_without_lift=sum(r['grasped'] and not r['lifted'] for r in rows),
            lifted_without_full_success=sum(r['lifted'] and not r['success'] for r in rows),
        ),
        observed_dx_m=[float(a[:, 0].min()), float(a[:, 0].max())],
        observed_dy_m=[float(a[:, 1].min()), float(a[:, 1].max())],
        observed_yaw_deg=[float(np.degrees(a[:, 2].min())), float(np.degrees(a[:, 2].max()))],
        grid_counts=total.tolist(), grid_successes=passed.tolist(),
        yaw_counts=yaw_total.tolist(), yaw_successes=yaw_passed.tolist(),
        sampling='Uniform random starts, grouped into bins after evaluation; not fixed grid points.',
    )
    assert result['successes'] + sum(result['failure_stages'].values()) == len(rows)
    results[key] = result
    rates = np.divide(passed, total, out=np.full((5, 5), np.nan), where=total > 0)
    extent = [-halfwidth * 100, halfwidth * 100] * 2
    im = ax.imshow(rates, origin='lower', vmin=0, vmax=1, cmap='RdYlGn', extent=extent)
    cell = halfwidth * 200 / 5
    for iy in range(5):
        for ix in range(5):
            ax.text(extent[0] + (ix+.5)*cell, extent[2] + (iy+.5)*cell,
                    f'{passed[iy, ix]}/{total[iy, ix]}', ha='center', va='center', fontsize=9,
                    color='white' if rates[iy, ix] < .12 else 'black')
    ax.set_title(title + f"\n{result['successes']}/{len(rows)} full successes")
    ax.set_xlabel('X offset from nominal pickup (cm)')
    ax.set_ylabel('Y offset from nominal pickup (cm)')
    fig.colorbar(im, ax=ax, shrink=.7, label='Full success fraction')
fig.suptitle('Frozen-policy workspace screening | seed42 | unchanged success rules\n'
             'Uniform random starts; cells show successes / trials, with yaw mixed within each cell', fontsize=13)
fig.savefig(ROOT / 'box_workspace_screening.png', dpi=170)
fig.savefig(ROOT / 'box_workspace_screening.svg')
(ROOT / 'box_workspace_screening_summary.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps({k: {'successes': v['successes'], 'summary': v['summary'],
                     'failure_stages': v['failure_stages']} for k, v in results.items()}, indent=2))
