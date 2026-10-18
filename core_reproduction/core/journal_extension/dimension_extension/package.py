"""Portable source, frozen scope, certificates and audit archive."""
from pathlib import Path
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parent
files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
       and p.suffix not in ('.pyc','.zip') and p.name!='RELEASE.json']
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
release=ROOT/'RELEASE.json'
release.write_text(json.dumps(dict(files=manifest,full_102_run=False,server_reproduced=False),indent=2)+'\n')
archive=ROOT.parent/'certified_dimension_extension_v1_bundle.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files+[release]:z.write(p,str(Path(ROOT.name)/p.relative_to(ROOT)))
print(json.dumps(dict(archive=str(archive),files=len(files)+1,bytes=archive.stat().st_size,
                     sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),indent=2))
