"""Package a self-contained server release; exclude all experimental outputs."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and
           not any(x in p.relative_to(ROOT).parts for x in ['results','__pycache__','.git']) and
           p.suffix!='.pyc' and p.name!='release_manifest.json']
    manifest={'files':[{'path':str(p.relative_to(ROOT)),
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (ROOT/'release_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    path=ROOT.parent/(ROOT.name+'.zip')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[ROOT/'release_manifest.json']:z.write(p,ROOT.name+'/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(path) as z:assert z.testzip() is None
    print(path);print('bytes',path.stat().st_size,'sha256',hashlib.sha256(path.read_bytes()).hexdigest())
if __name__=='__main__':main()
