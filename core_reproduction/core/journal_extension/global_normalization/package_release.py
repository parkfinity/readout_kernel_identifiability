"""Package code and the explicitly selected final local run without caches."""
import argparse, hashlib, json, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def pack(target,base,files):
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(base)))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    return {'path':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);a=parser.parse_args()
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and
           not any(x in p.parts for x in ('results','__pycache__')) and p.name!='.DS_Store']
    records=[pack(ROOT.parent/'global_pauli_reparameterization_v1_code.zip',ROOT.parent,files)]
    files=[p for p in sorted(a.run.rglob('*')) if p.is_file()]
    records.append(pack(ROOT.parent/'global_pauli_local_results.zip',a.run.parent,files))
    print(json.dumps(records,indent=2))
