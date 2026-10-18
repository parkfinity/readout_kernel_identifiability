"""Package the local control and its unchanged, minimal replay dependencies."""
from pathlib import Path
import json,zipfile,hashlib

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'readout_product_baseline_v1_bundle.zip'
files=[]
for folder in ['readout_product_baseline_v1','readout_metric_shot_design_v2']:
    for p in (ROOT/folder).rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        if folder=='readout_metric_shot_design_v2' and 'results' in p.parts:continue
        if p.suffix in ['.py','.txt','.json','.sh','.npz']:files.append(p)
old=ROOT/'readout_metric_shot_v2_server_audit/results/full_20261005T050625_181530Z'
files+=list(old.glob('panel*_hub_zz_point*_samples.npz'))
assert len(list(old.glob('panel*_hub_zz_point*_samples.npz')))==15
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(files):z.write(p,'readout_product_baseline_v1_bundle/'+str(p.relative_to(ROOT)))
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    z.writestr('readout_product_baseline_v1_bundle/BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(OUT) as z:assert z.testzip() is None
print(OUT,'files',len(files)+1,'bytes',OUT.stat().st_size)
