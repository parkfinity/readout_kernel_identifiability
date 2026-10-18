"""Build a portable input+code+original-results bundle, excluding caches."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'results/pilot_20261001T161006_106481Z'


def main():
    dest=ROOT/'frozen_inputs';dest.mkdir(exist_ok=True)
    for f in (RUN/'inputs').glob('*.json'): shutil.copy2(f,dest/f.name)
    scope=json.loads((RUN/'run.json').read_text())['scope']
    (dest/'scope.json').write_text(json.dumps(scope,indent=2)+'\n')
    chosen=[]
    for f in sorted(ROOT.rglob('*')):
        if not f.is_file() or '__pycache__' in f.parts or f.suffix=='.pyc': continue
        rel=f.relative_to(ROOT)
        if rel.parts[0]=='results' and rel.parts[1]!=RUN.name: continue
        if f.name=='RELEASE.json' or f.suffix=='.zip': continue
        chosen.append(f)
    manifest={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in chosen}
    release=ROOT/'RELEASE.json'
    release.write_text(json.dumps({'files':manifest,'original_run':RUN.name,'full_102_run':False},indent=2)+'\n')
    archive=ROOT.parent/'certified_dimension_pilot_v1_bundle.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in chosen+[release]: z.write(f,str(Path(ROOT.name)/f.relative_to(ROOT)))
    print(json.dumps({'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                      'files':len(chosen)+1,'bytes':archive.stat().st_size},indent=2))


if __name__=='__main__':main()
