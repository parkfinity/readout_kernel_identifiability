"""Maintainer packaging. Never include local results as apparent server runs."""
import zipfile
from core import *

def main():
    load_plan(verify_package=False)
    paths=sorted(p for p in HERE.rglob('*') if p.is_file() and
        'results' not in p.parts and '__pycache__' not in p.parts and
        p.name!='PACKAGE_MANIFEST.json' and p.suffix not in ('.zip','.pyc'))
    save(HERE/'PACKAGE_MANIFEST.json',{str(p.relative_to(HERE)):sha(p) for p in paths})
    paths.append(HERE/'PACKAGE_MANIFEST.json')
    out=HERE.parent/'readout_metric_shot_design_v2_bundle.zip'
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,Path(HERE.name)/p.relative_to(HERE))
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        assert not any('/results/' in name for name in z.namelist())
    print(json.dumps(dict(archive=str(out),bytes=out.stat().st_size,files=len(paths),sha256=sha(out)),indent=2))

if __name__=='__main__':main()
