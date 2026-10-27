"""Repair paths and flattened interpreter links after this workspace's PC transfer."""
from pathlib import Path
import json

root = Path('/mnt/d/work/robotics/IsaacLabTutorial')
venv = root / '.venv'
old = '/mnt/e/robotics/IsaacLabTutorial'
for name, target in [('python', '/usr/bin/python3.12'), ('python3', 'python'), ('python3.12', 'python')]:
    path = venv / 'bin' / name
    if path.is_file() and path.stat().st_size == 0:
        path.unlink()
        path.symlink_to(target)
        print(f'Repaired {path} -> {target}')
lib64 = venv / 'lib64'
if lib64.is_file() and lib64.stat().st_size == 0:
    lib64.unlink()
    lib64.symlink_to('lib', target_is_directory=True)
    print('Repaired lib64 -> lib')
changed = []
for path in (venv / 'bin').iterdir():
    if not path.is_file() or path.is_symlink():
        continue
    raw = path.read_bytes()
    if b'\0' in raw:
        continue
    try:
        text = raw.decode()
    except UnicodeDecodeError:
        continue
    if old in text:
        path.write_text(text.replace(old, str(root)))
        changed.append(path.name)
pth = venv / 'lib/python3.12/site-packages/isaaclab_tutorial.pth'
pth.write_text(str(root / 'src') + '\n')
direct = venv / 'lib/python3.12/site-packages/isaaclab_tutorial-0.1.0.dist-info/direct_url.json'
if direct.exists():
    metadata = json.loads(direct.read_text())
    metadata['url'] = root.as_uri()
    direct.write_text(json.dumps(metadata))
print(f'Updated {len(changed)} launch/activation scripts and editable package path')
