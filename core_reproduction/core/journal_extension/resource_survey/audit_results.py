"""Read-back integrity and preserved-array audit; no timing replay."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def audit(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    for entry in manifest['files']:
        p=folder/entry['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
    d=json.loads((folder/'summary.json').read_text())
    assert d['passed'] and len(d['rows'])==120
    assert sum(r['status']=='DUPLICATE' for r in d['rows'])==18
    count=0;maxima={}
    for r in d['rows']:
        if r['status']=='DUPLICATE':continue
        assert r['status'].startswith('PASS')
        assert r['validation']['passed']
        for k,v in r['validation']['max_errors'].items():
            assert v<=r['validation']['limits'][k]
            maxima[k]=max(maxima.get(k,0),v)
        with np.load(folder/r['id']/'validation_arrays.npz',allow_pickle=False) as a:
            for k in a.files:assert np.isfinite(a[k]).all();count+=1
            for seed in (812,1907):
                for suffix,tol in [('K',1e-10),('kernel_jacobian',1e-9),('theta_final',1e-9),('prediction',1e-9)]:
                    x,y=(a[f'{seed}_{m}_{suffix}'] for m in ('boundary','transport'))
                    assert x.shape==y.shape
                    assert not x.size or np.max(abs(x-y))<=tol,(r['id'],suffix)
        b,t=(r['metrics'][m] for m in ('boundary','transport'))
        assert r['delta']=={k:b[k]-t[k] for k in b}
    return {'passed':True,'manifest_files':len(manifest['files']),'arrays_checked':count,
            'slots_checked':102,'max_recorded_errors':maxima}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args()
    print(json.dumps(audit(args.folder),indent=2))
