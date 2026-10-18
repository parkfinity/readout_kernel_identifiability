"""Audit local repeatability and create a self-contained server bundle."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import sympy as sp
from design import HERE

def strip_time(obj):
    if isinstance(obj,dict): return {k:strip_time(v) for k,v in obj.items() if k!='elapsed_seconds'}
    if isinstance(obj,list): return [strip_time(v) for v in obj]
    return obj

def compare(a,b):
    if isinstance(a,dict): return a.keys()==b.keys() and all(compare(a[k],b[k]) for k in a)
    if isinstance(a,list): return len(a)==len(b) and all(compare(x,y) for x,y in zip(a,b))
    if isinstance(a,float): return bool(np.isclose(a,b,rtol=1e-10,atol=1e-12))
    return a==b

def run():
    dirs=sorted(p.parent for p in (HERE/'results').glob('*/summary.json'))
    latest=dirs[-1]
    checks={}
    expected=['metric_summary.json','readout_summary.json','finite_shots.json',
              'exact_controls/summary.json','certificate_recheck.json',
              'strong_baseline_followup/one_basis_baselines.json',
              'strong_baseline_followup/finite_shot_baselines.json','correlation_control.json']
    if len(dirs)>=2:
        for name in expected:
            a=strip_time(json.loads((dirs[-2]/name).read_text()));b=strip_time(json.loads((dirs[-1]/name).read_text()))
            checks['two_local_environments_'+name]=compare(a,b)
    corr=json.loads((latest/'correlation_control.json').read_text())
    for w in corr['witnesses']:
        checks['correlation_minor_'+str(w['prime'])]=(int(sp.Matrix(w['minor_matrix']).det())%w['prime']==w['determinant_mod_prime']!=0)
    rows=json.loads((latest/'metric_summary.json').read_text())
    checks['all_metric_rank_tolerances_stable']=all(len(set(s['sensitivity_ranks']))==1 for r in rows for a in r['arms'].values() for s in a['validation'])
    checks['all_metric_heldout_ranks_stable']=all(len(set(s['rank'] for s in a['validation']))==1 for r in rows for a in r['arms'].values())
    checks['20_full_rank_certificates']=json.loads((latest/'exact_controls/summary.json').read_text())['exact_full_rank_count']==20
    checks['engine_copy_unmodified']=(HERE/'vendor/fourier_gram.py').read_bytes()==(HERE.parent/'fourier_gram_identifiability_v1/experiment/fourier_gram.py').read_bytes()
    checks['exact_engine_copy_unmodified']=(HERE/'vendor/copied_from_extension_v1/exact_engine.py').read_bytes()==(HERE.parent/'fourier_gram_identifiability_v1/experiment/copied_from_extension_v1/exact_engine.py').read_bytes()
    audit=dict(passed=all(checks.values()),checks=checks,local_runs=[p.name for p in dirs],
               environments=[json.loads((p/'environment.json').read_text()) for p in dirs],
               scope='Two local Python environments, not a Linux server or hardware replay')
    (HERE/'LOCAL_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    assert audit['passed'],checks
    with (HERE/'METRIC_RESULTS.tsv').open('w') as f:
        wr=csv.writer(f,delimiter='\t');wr.writerow(['setting','K','features','bases','identity','fixed_diagonal','random_dense','selected_dense','centered_feature_rank'])
        for r in rows:
            wr.writerow([r['setting'],r['K'],len(r['observables']),len(r['basis_schedule'])]+[r['arms'][a]['validation'][0]['rank'] for a in ['identity','fixed_diagonal','random_dense','selected_dense']]+[r['centered_feature_rank_diagnostics'][0]['rank']])
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.zip') and p.name!='MANIFEST.json')
    manifest={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (HERE/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files.append(HERE/'MANIFEST.json')
    target=HERE.parent/'readout_metric_design_v1_bundle.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,Path(HERE.name)/p.relative_to(HERE))
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    print(json.dumps(dict(audit_passed=True,archive=str(target),files=len(files),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':run()
