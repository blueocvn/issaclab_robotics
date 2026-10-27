"""One-off snapshot export into the user's destination clone, preserving its Git history."""
import hashlib
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

source = Path('/mnt/d/work/robotics/IsaacLabTutorial')
target = Path('/mnt/d/work/robotics/issaclab_robotics')
assert source != target and (target/'.git').is_dir()
assert subprocess.check_output(['git','-C',str(target),'remote','get-url','origin'], text=True).strip() == \
    'https://github.com/blueocvn/issaclab_robotics.git'
tracked = subprocess.check_output(['git','-C',str(source),'ls-files','--cached','--others','--exclude-standard','-z'])
paths = {Path(p.decode()) for p in tracked.split(b'\0') if p}
for path in (source/'logs').rglob('*'):
    if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.pyc','.tmp','.temp','.diff'):
        continue
    if path.name in ('owner.lock','STOP','NO_SHUTDOWN') or 'git' in path.relative_to(source/'logs').parts:
        continue
    paths.add(path.relative_to(source))
records = []
models = []
for relative in sorted(paths):
    if relative.parts[0] in ('.git','.venv') or relative.name.startswith('.env') and relative.name != '.env.example':
        continue
    src, dst = source/relative, target/relative
    assert src.is_file() and src.resolve().is_relative_to(source.resolve())
    assert dst.resolve().is_relative_to(target.resolve()) and '.git' not in relative.parts
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    digest = hashlib.sha256(dst.read_bytes()).hexdigest()
    item = dict(path=relative.as_posix(), bytes=dst.stat().st_size, sha256=digest)
    records.append(item)
    if relative.suffix == '.pt':
        models.append(dict(item, kind='training_checkpoint' if re.fullmatch(r'model_\d+\.pt', relative.name)
                           else ('reset_dataset' if relative.name == 'reset_poses.pt' else 'exported_policy')))
manifest = dict(created_at=datetime.now(UTC).isoformat(),
                source_repository='https://github.com/isaac-sim/IsaacLabTutorial.git',
                source_base_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                destination_repository='https://github.com/blueocvn/issaclab_robotics',
                note='Working-tree snapshot including uncommitted code and explicit training archive; existing destination history retained.',
                files=records)
(target/'publication_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
(target/'checkpoint_manifest.json').write_text(json.dumps(dict(models=models), indent=2)+'\n')
archive_paths=[r['path'] for r in records if r['path'].startswith('logs/')]
(target/'.git'/'archive_paths.nul').write_bytes(b''.join(p.encode()+b'\0' for p in archive_paths))
print(json.dumps(dict(files=len(records), training_checkpoints=sum(r['kind']=='training_checkpoint' for r in models),
                      pytorch_artifacts=len(models), total_bytes=sum(r['bytes'] for r in records)), indent=2))
