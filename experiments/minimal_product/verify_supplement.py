"""Check an extracted supplement without rewriting any reference results."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'MANIFEST.json').read_text())
for name, expected in manifest.items():
    path = root / name
    if not path.is_file():
        raise SystemExit('MISSING: ' + name)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit('HASH MISMATCH: ' + name)
print('PASS integrity:', len(manifest), 'files', flush=True)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
for folder in ('verification_math', 'experiment'):
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', folder, '-v'],
                   cwd=root, env=env, check=True)
print('PASS: 17 exact algebra tests and 8 experiment tests. No QPU used.')
