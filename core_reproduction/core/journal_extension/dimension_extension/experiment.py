"""Fixed 30-setting extension; bounded identities, feasible witnesses, resource audit."""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import signal
import time
import numpy as np
import sympy as sp
import mpmath
from mpmath import iv
import exact_engine as ex
import interval_engine as ie
from normalizer import normalize, delete_common_isometries, resources, signed_word

ROOT=Path(__file__).resolve().parent
MODULES=('yz_cx_encoding_circuit','hubregtsen_encoding_circuit','kyriienko_nonlinear_encoding_circuit')
READOUTS=('local_xyz','local_z','pauli_weight_le_2')

def save(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def names():
    return [(m,n,l,r,'%s_n%d_l%d_%s'%(m,n,l,r)) for m in MODULES for n in (2,4)
            for l in ((1,) if m.startswith('kyriienko') else (1,2)) for r in READOUTS]

def freeze(source):
    dest=ROOT/'frozen_inputs'
    if dest.exists(): raise ValueError('Frozen inputs already exist; do not overwrite')
    dest.mkdir(); slots=[]
    for m,n,l,r,name in names():
        p=source/name/'programs.json'; info=json.loads(p.read_text())['compiler'];ex.validate(info)
        old=json.loads((p.parent/'result.json').read_text())
        save(dest/(name+'.json'),info)
        slots.append(dict(id=name,module=m,n=n,layers=l,readout=r,file_sha256=sha(dest/(name+'.json')),
                          source_sha256=ie.info_hash(info),archive_sha256=sha(p),
                          old_inactive=sorted(set(range(info['count']))-set(old['unknown_symbols']))))
    save(dest/'scope.json',slots)
    print('Frozen',len(slots),'settings,',sha(dest/'scope.json'))

def bounded_identity(info):
    def expired(*args): raise TimeoutError('20 second symbolic budget exceeded')
    old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,20)
    try:
        try: return ex.certify(info)[0],None
        finally: signal.setitimer(signal.ITIMER_REAL,0)
    except (TimeoutError,RuntimeError) as e: return None,str(e)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)

def minor(rows,target):
    if not target:return dict(lower=0,rows=[],columns=[],determinant_bounds=['1','1'])
    a=np.array([[ie.mid(v) for v in row] for row in rows]);rr=list(range(len(rows)));cc=list(range(a.shape[1]));rank=0
    for k in range(min(target,*a.shape)):
        i,j=np.unravel_index(np.argmax(abs(a[k:,k:])),a[k:,k:].shape);i+=k;j+=k
        if abs(a[i,j])<1e-10:break # candidate selection only
        a[[k,i],:]=a[[i,k],:];a[:,[k,j]]=a[:,[j,k]]
        rr[k],rr[i]=rr[i],rr[k];cc[k],cc[j]=cc[j],cc[k]
        for r in range(k+1,a.shape[0]):a[r,k+1:]-=a[r,k]/a[k,k]*a[k,k+1:]
        rank+=1
    for q in range(rank,0,-1):
        d=ie.determinant([[rows[r][c] for c in cc[:q]] for r in rr[:q]])
        if d is not None and ie.nonzero(d):return dict(lower=q,rows=rr[:q],columns=cc[:q],determinant_bounds=ie.bounds(d))
    return dict(lower=0,rows=[],columns=[],determinant_bounds=['1','1'])

def commute_prune(info):
    """Known observable-commutant rule, valid even for input-dependent angles."""
    out=copy.deepcopy(info);removed=[];p=out['program']
    for i,g in enumerate(p):
        if g['op']!='R':continue
        w=g['pauli'];safe=True
        for h in p[i+1:]:
            if h['op']=='R':
                if not ex.commutes(w,h['pauli']):safe=False;break
            else:w,_=signed_word(w,[h],info['n'],'s')
        if safe and all(ex.commutes(w,o) for o in info['observables']):removed.append(i)
    out['program']=[g for i,g in enumerate(p) if i not in removed]
    return out

def optimize(info,commutant=False):
    out=copy.deepcopy(info)
    for _ in range(50):
        before=out['program'];out,_=normalize(out);out,_=delete_common_isometries(out)
        if commutant:out=commute_prune(out)
        out,_=normalize(out)
        if before==out['program']:return out
    raise RuntimeError('Optimizer failed to reach fixed point')

def zero_coordinates(info,inactive):
    out=copy.deepcopy(info)
    for g in out['program']:
        if g['op']=='R':g['terms']=[v for v in g['terms'] if v[1] not in inactive]
    out['program']=[g for g in out['program'] if g['op']!='R' or g['terms']]
    return out

def resource_audit(info,inactive):
    g=optimize(info);b=optimize(g,True);a=optimize(zero_coordinates(b,inactive),True)
    arms=dict(global_isometry=g,global_plus_commutant=b,certified_zero=a)
    metrics={k:resources(v) for k,v in arms.items()}
    for v in metrics.values():v['two_shift_feature_evaluations_per_input']=2*v['trainable_angle_tasks']
    return dict(programs=arms,metrics=metrics,
                incremental={k:metrics['global_plus_commutant'][k]-metrics['certified_zero'][k] for k in metrics['certified_zero']})

def run(resume=None):
    slots=json.loads((ROOT/'frozen_inputs/scope.json').read_text())
    assert {s['id'] for s in slots}=={v[-1] for v in names()} and len(slots)==30
    out=resume or ROOT/'results'/('extension_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    if resume:
        assert not (out/'manifest.json').exists(),'Completed runs cannot be resumed'
        assert json.loads((out/'run.json').read_text())['scope']==slots
        shutil.copy2(Path(__file__),out/'source_snapshot/experiment_resume.py')
        save(out/'resume.json',dict(utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(Path(__file__)),
             reason='Disable SIGALRM before handling a simultaneous symbolic-budget exception; retain completed slots unchanged',
             retained_files={p.name:sha(p) for p in sorted(out.glob('*_n*.json'))}))
    else:
        (out/'source_snapshot').mkdir(parents=True);shutil.copytree(ROOT/'frozen_inputs',out/'inputs')
        for p in ROOT.iterdir():
            if p.suffix in ('.py','.md','.txt','.sh'):shutil.copy2(p,out/'source_snapshot'/p.name)
        save(out/'run.json',dict(scope=slots,seed=20261003,pair_budget=24,precision_dps=50,symbolic_timeout_seconds=20,
                            environment=dict(python=platform.python_version(),numpy=np.__version__,sympy=sp.__version__,mpmath=mpmath.__version__,platform=platform.platform()),
                            full_102_run=False,protocol_sha256=sha(ROOT/'PROTOCOL_KO.md')))
    results=[];start=time.perf_counter()
    for slot in slots:
        if resume and (out/(slot['id']+'.json')).exists():
            results.append(json.loads((out/(slot['id']+'.json')).read_text()));continue
        tick=time.perf_counter();p=out/'inputs'/(slot['id']+'.json');assert sha(p)==slot['file_sha256']
        info=json.loads(p.read_text());ex.validate(info)
        proof,reason=bounded_identity(info);symbolic_seconds=time.perf_counter()-tick
        inactive=proof['inactive'] if proof else [];upper=proof['r_upper'] if proof else info['count']
        rng=random.Random(20261003);theta=[str(F(rng.randint(2,24),17)) for _ in range(info['count'])]
        pairs=[([str(F(rng.randint(-15,15),19)) for _ in range(2)],[str(F(rng.randint(-15,15),19)) for _ in range(2)]) for _ in range(24)]
        iv.dps=50;rows=[];active={};distances=[]
        for j,(x,xp) in enumerate(pairs):
            d,grad=ie.derivative(info,theta,x,xp);rows.append(grad);distances.append(ie.bounds(d))
            for k,v in enumerate(grad):
                if ie.nonzero(v):
                    if k in inactive:raise RuntimeError('Conflicting certificates '+slot['id'])
                    active.setdefault(k,dict(parameter=k,pair_index=j,space='original_inputs',source_sha256=ie.info_hash(info),theta=theta,x=x,xprime=xp,dps=50,derivative_bounds=ie.bounds(v)))
            m=minor(rows,upper)
            if j>=1 and m['lower']==upper and len(active)+len(inactive)==info['count']:break
        status=['EXACT_INACTIVE' if k in inactive else 'ACTIVE_CERTIFIED' if k in active else 'UNDECIDED' for k in range(info['count'])]
        r=dict(slot,parameter_count=info['count'],exact=proof,symbolic_failure=reason,r_upper=upper,minor=m,
               status=status,counts=dict(Counter(status)),active_witnesses=list(active.values()),theta=theta,pairs=pairs[:len(rows)],
               derivative_intervals=[[ie.bounds(v) for v in row] for row in rows],distance_intervals=distances,
               new_inactive=sorted(set(inactive)-set(slot['old_inactive'])),old_not_recovered=sorted(set(slot['old_inactive'])-set(inactive)),
               resources=resource_audit(info,inactive),symbolic_seconds=symbolic_seconds,elapsed_seconds=time.perf_counter()-tick)
        save(out/(slot['id']+'.json'),r);results.append(r)
        print(slot['id'],r['counts'],'rank',m['lower'],upper,'extra',r['resources']['incremental'],'symbolic',reason,flush=True)
    summary=dict(slots=len(results),source_circuit_configurations=10,symbols=sum(r['parameter_count'] for r in results),
                 counts=dict(sum((Counter(r['status']) for r in results),Counter())),exact_completed=sum(r['exact'] is not None for r in results),
                 rank_closed_slots=sum(r['minor']['lower']==r['r_upper'] for r in results),new_inactive=sum(len(r['new_inactive']) for r in results),
                 old_not_recovered=sum(len(r['old_not_recovered']) for r in results),
                 resource_positive_slots=[r['id'] for r in results if r['resources']['incremental']['trainable_angle_tasks']>0],
                 completed_slot_seconds=sum(r['elapsed_seconds'] for r in results),final_execution_segment_seconds=time.perf_counter()-start,
                 resumed=bool(resume),full_102_run=False,hardware_or_timing_savings_measured=False)
    save(out/'summary.json',summary)
    save(out/'manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps(dict(output=str(out),summary=summary),indent=2))

def replay(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,h in manifest.items():assert sha(folder/name)==h,name
    run=json.loads((folder/'run.json').read_text());checked=0;witnesses=0;minors=0
    for slot in run['scope']:
        info=json.loads((folder/'inputs'/(slot['id']+'.json')).read_text());r=json.loads((folder/(slot['id']+'.json')).read_text())
        assert ie.info_hash(info)==slot['source_sha256'];proof=r['exact']
        if proof is not None:assert ex.certify(info)[0]==proof
        inactive=proof['inactive'] if proof else []
        assert r['r_upper']==(proof['r_upper'] if proof else info['count'])
        iv.dps=65;needed=set(r['minor']['rows'])|{w['pair_index'] for w in r['active_witnesses']};cache={}
        for j in needed:cache[j]=ie.derivative(info,r['theta'],*r['pairs'][j])[1]
        active=set()
        for w in r['active_witnesses']:
            assert w['space']=='original_inputs' and w['theta']==r['theta'] and w['source_sha256']==ie.info_hash(info)
            assert [w['x'],w['xprime']]==r['pairs'][w['pair_index']]
            assert ie.nonzero(cache[w['pair_index']][w['parameter']]);active.add(w['parameter']);witnesses+=1
        assert not active.intersection(inactive)
        statuses=['EXACT_INACTIVE' if k in inactive else 'ACTIVE_CERTIFIED' if k in active else 'UNDECIDED' for k in range(info['count'])]
        assert statuses==r['status'] and dict(Counter(statuses))==r['counts']
        m=r['minor'];assert m['lower']<=r['r_upper']
        if m['lower']:
            assert len(m['rows'])==len(m['columns'])==m['lower']
            d=ie.determinant([[cache[i][j] for j in m['columns']] for i in m['rows']]);assert d is not None and ie.nonzero(d);minors+=1
        assert resource_audit(info,inactive)==r['resources'];checked+=1
    return dict(passed=True,slots=checked,active_witnesses=witnesses,nonzero_minors=minors,precision_dps=65,manifest_files=len(manifest),manifest_sha256=sha(folder/'manifest.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path);p.add_argument('--replay',type=Path);p.add_argument('--audit-output',type=Path);p.add_argument('--resume',type=Path);a=p.parse_args()
    if a.freeze:freeze(a.freeze)
    elif a.replay:
        result=replay(a.replay)
        if a.audit_output:save(a.audit_output,result)
        print(json.dumps(result,indent=2))
    else:run(a.resume)
