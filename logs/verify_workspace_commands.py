"""Dry-run all user audit populations and verify launch dimensions."""
import contextlib
import io
import json
from pathlib import Path

from isaaclab_tutorial.tasks.place_box.workspace_audit import main

root = Path('logs')
teacher = 'logs/rsl_rl/so101_box_state/2026-10-26_14-59-07_so101-box-release-v3-refine/model_2196.pt'
summary = {}
for population in ('grid', 'random', 'stage', 'nominal', 'regression'):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        assert main(['--checkpoint', teacher, '--policy', 'teacher', '--population', population, '--dry-run']) == 0
    data = json.loads(stream.getvalue())
    jobs = data['jobs']
    assert len({j['sampling_seed'] for j in jobs}) == len(jobs)
    assert all('env.events.reset_home.params.nominal_fraction=0.0' in j['command'] for j in jobs)
    (root/f'workspace_{population}_dry_run.json').write_text(json.dumps(data, indent=2)+'\n')
    summary[population] = dict(jobs=len(jobs), episodes=sum(j['count'] for j in jobs),
                               distinct_sampling_seeds=len({j['sampling_seed'] for j in jobs}))
assert summary['grid']['episodes'] == 3000
assert summary['random']['episodes'] == summary['nominal']['episodes'] == 3072
assert summary['regression']['episodes'] == 3
(root/'workspace_commands_verification.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
