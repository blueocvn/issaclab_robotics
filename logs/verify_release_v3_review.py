import json
import math
import os
import re
from pathlib import Path

log = Path('logs/local_pc_release_v3_integration.log').read_text()
iterations = re.findall(r'Learning iteration (\d+)/\d+', log)
assert iterations == ['1598', '1599'], iterations
for name in ('value', 'surrogate', 'entropy'):
    values = [float(v) for v in re.findall(r'Mean ' + name + r' loss:\s*([^\s]+)', log)]
    assert len(values) == 2 and all(math.isfinite(v) for v in values), (name, values)
for name in ('on_target', 'released', 'withdrawn', 'upright'):
    assert log.count('Metrics/' + name + '_rate:') == 2
audit = json.loads(Path('logs/local_pc_release_v3_smoke_audit.json').read_text())
rows = audit['episodes']
assert len(rows) == 8 and {r['env_id'] for r in rows} == set(range(8))
for name in ('on_target', 'released', 'withdrawn', 'upright'):
    assert all(isinstance(r[name], bool) for r in rows)
    assert audit['summary'][name + '_rate'] == sum(r[name] for r in rows) / 8
assert all(r['grasped'] and r['lifted'] for r in rows)
active = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        comm = (proc / 'comm').read_text().strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if comm.startswith('python') and ('isaaclab_tutorial.tasks.place_box' in command or '/.venv/bin/isaaclab ' in command or '/.venv/bin/pytest ' in command):
        active.append(command)
assert not active, active
print(json.dumps(dict(integration_updates=len(iterations), finite_losses=True,
                      terminal_stage_schema_valid=True, audit=audit['summary'],
                      verification_processes_remaining=active), sort_keys=True))
