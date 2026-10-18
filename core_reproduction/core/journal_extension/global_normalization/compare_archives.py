"""Read-only identity comparison of the two already completed server studies."""
import argparse, hashlib, json, zipfile
from pathlib import Path


def run(full,survey):
    rows=[]
    with zipfile.ZipFile(full) as a,zipfile.ZipFile(survey) as b:
        for n in (2,4):
            name='chebyshev_rx_n%d_l2'%n
            pa=next(p for p in a.namelist() if p.endswith('/validation/'+name+'.json'))
            pb=next(p for p in b.namelist() if p.endswith('/'+name+'_local_xyz/programs.json'))
            va=json.loads(a.read(pa));vb=json.loads(b.read(pb))
            for method in ('boundary','transport'):
                ia=next(r['info'] for r in va['methods'] if r['method']==method);ib=vb[method]
                checks={k:ia[k]==ib[k] for k in ('n','count','program','ids','data_expressions','observables','weights')}
                rows.append({'config':name,'method':method,'checks':checks,'passed':all(checks.values())})
    return {'passed':all(r['passed'] for r in rows),'rows':rows,
            'archives':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (full,survey)],
            'scope':'Saved final executable IR and retained symbol identities, not identical proof traces or timers'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('full',type=Path);p.add_argument('survey',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();r=run(a.full,a.survey)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2));raise SystemExit(0 if r['passed'] else 1)
