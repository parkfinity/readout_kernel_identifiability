"""Freeze, run, and replay the 24-slot restricted certificate pilot."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import time
import numpy as np
import mpmath
from mpmath import iv
import sympy as sp
import exact_engine as ex
import interval_engine as ie

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT.parent/'cross_family_resource_v1/results/survey_20261001T085953_343158Z'


def save(p, v): p.write_text(json.dumps(v, indent=2, ensure_ascii=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def propose_minor(rows, target):
    a = np.array([[ie.mid(v) for v in row] for row in rows], dtype=float)
    order_r = list(range(a.shape[0])); order_c = list(range(a.shape[1])); rank = 0
    for k in range(min(target, *a.shape)):
        i,j = np.unravel_index(np.argmax(abs(a[k:,k:])), a[k:,k:].shape); i += k; j += k
        if abs(a[i,j]) < 1e-10: break  # candidate selection ONLY
        a[[k,i],:] = a[[i,k],:]; a[:,[k,j]] = a[:,[j,k]]
        order_r[k],order_r[i] = order_r[i],order_r[k]
        order_c[k],order_c[j] = order_c[j],order_c[k]
        for r in range(k+1, a.shape[0]): a[r,k+1:] -= (a[r,k]/a[k,k])*a[k,k+1:]
        rank += 1
    for q in range(rank, 0, -1):
        rr, cc = order_r[:q],order_c[:q]
        d = ie.determinant([[rows[r][c] for c in cc] for r in rr])
        if d is not None and ie.nonzero(d): return {'lower':q,'rows':rr,'columns':cc,'determinant_bounds':ie.bounds(d)}
    return {'lower':0,'rows':[],'columns':[],'determinant_bounds':['1','1']}


def freeze(source, out):
    slots = []
    for module in ('chebyshev_rx','param_z_feature_map'):
        for n in (2,4):
            for layers in (1,2):
                for readout in ('local_xyz','local_z','pauli_weight_le_2'):
                    name = '%s_n%d_l%d_%s' % (module,n,layers,readout)
                    p = source/name/'programs.json'; info = json.loads(p.read_text())['compiler']; ex.validate(info)
                    dest = out/'inputs'/(name+'.json'); save(dest, info)
                    row = json.loads((p.parent/'result.json').read_text())
                    slots.append({'id':name,'module':module,'n':n,'layers':layers,'readout':readout,
                                  'file_sha256':sha(dest),'source_sha256':ie.info_hash(info),
                                  'archive_file_sha256':sha(p), 'old_inactive':sorted(set(range(info['count']))-set(row['unknown_symbols']))})
    return slots


def run(source):
    out = ROOT/'results'/('pilot_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    (out/'inputs').mkdir(parents=True); (out/'source_snapshot').mkdir()
    for f in list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+list(ROOT.glob('*.txt'))+list(ROOT.glob('*.sh')):
        shutil.copy2(f,out/'source_snapshot'/f.name)
    slots = freeze(source,out)
    env = {'python':platform.python_version(),'sympy':sp.__version__,'numpy':np.__version__,
           'mpmath':mpmath.__version__,'platform':platform.platform()}
    save(out/'run.json',{'scope':slots,'environment':env,'source_archive':str(source),
                        'precision_dps':50,'pair_budget':24,'seed':20261002,
                        'semantics':'exact integer IR coefficients; supported data expressions; no angle snapping',
                        'full_102_run':False,'probabilistic_identity_tests':False})
    results = []; start = time.perf_counter()
    for slot in slots:
        tick = time.perf_counter(); info = json.loads((out/'inputs'/(slot['id']+'.json')).read_text())
        proof, d = ex.certify(info); symbolic_seconds = time.perf_counter()-tick
        rng = random.Random(20261002)
        theta = [str(F(rng.randint(2,24),17)) for _ in range(info['count'])]
        pairs = [([str(F(rng.randint(-15,15),19)) for _ in range(2)],
                  [str(F(rng.randint(-15,15),19)) for _ in range(2)]) for _ in range(24)]
        iv.dps = 50; rows = []; active = {}; distances = []; witness_tick = time.perf_counter()
        for row, (x,xp) in enumerate(pairs):
            distance, grad = ie.derivative(info,theta,x,xp); rows.append(grad); distances.append(ie.bounds(distance))
            for k,v in enumerate(grad):
                if ie.nonzero(v) and k not in active:
                    if k in proof['inactive']: raise RuntimeError('Certificate conflict '+slot['id'])
                    active[k] = {'parameter':k,'pair_index':row,'space':'original_inputs',
                                 'source_sha256':ie.info_hash(info),'theta':theta,'x':x,'xprime':xp,
                                 'dps':50,'derivative_bounds':ie.bounds(v)}
        minor = propose_minor(rows,proof['r_upper'])
        if minor['lower'] > proof['r_upper']: raise RuntimeError('Rank conflict')
        status = ['EXACT_INACTIVE' if k in proof['inactive'] else 'ACTIVE_CERTIFIED' if k in active else 'UNDECIDED'
                  for k in range(info['count'])]
        old = set(slot['old_inactive']); new = set(proof['inactive'])
        result = dict(slot, parameter_count=info['count'], exact=proof, minor=minor, status=status,
                      counts=dict(Counter(status)),active_witnesses=list(active.values()),
                      new_inactive=sorted(new-old),old_not_recovered=sorted(old-new),theta=theta,pairs=pairs,
                      derivative_intervals=[[ie.bounds(v) for v in row] for row in rows],distance_intervals=distances,
                      symbolic_seconds=symbolic_seconds,witness_seconds=time.perf_counter()-witness_tick,
                      elapsed_seconds=time.perf_counter()-tick)
        save(out/(slot['id']+'.json'),result); results.append(result)
        print(slot['id'],result['counts'],'rank',minor['lower'],proof['r_upper'],flush=True)
    summary = {'slots':len(results),'source_circuit_configurations':8,
               'symbols':sum(r['parameter_count'] for r in results),
               'counts':dict(sum((Counter(r['status']) for r in results),Counter())),
               'rank_closed_slots':sum(r['minor']['lower']==r['exact']['r_upper'] for r in results),
               'new_inactive':sum(len(r['new_inactive']) for r in results),
               'old_not_recovered':sum(len(r['old_not_recovered']) for r in results),
               'constant_kernel_slots':[r['id'] for r in results if r['exact']['distance_is_zero']],
               'elapsed_seconds':time.perf_counter()-start,'full_102_run':False,
               'new_gate_shot_savings_measured':False}
    save(out/'summary.json',summary)
    save(out/'manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps({'output':str(out),'summary':summary},indent=2))


def audit(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    for name, digest in manifest.items():
        if sha(folder/name) != digest: raise ValueError('Manifest mismatch '+name)
    run = json.loads((folder/'run.json').read_text()); checked=0; active_count=0; minor_count=0
    for slot in run['scope']:
        info = json.loads((folder/'inputs'/(slot['id']+'.json')).read_text())
        record = json.loads((folder/(slot['id']+'.json')).read_text())
        if ie.info_hash(info) != slot['source_sha256']: raise ValueError('IR hash mismatch')
        proof,_ = ex.certify(info)
        assert proof == record['exact']
        iv.dps = 65 # independent replay at higher precision
        theta = record['theta']; cache={}
        needed = sorted(set(record['minor']['rows']) | {w['pair_index'] for w in record['active_witnesses']})
        for i in needed:
            x,xp = record['pairs'][i]; cache[i] = ie.derivative(info,theta,x,xp)[1]
        for w in record['active_witnesses']:
            assert w['space']=='original_inputs' and w['theta']==theta and w['source_sha256']==ie.info_hash(info)
            assert [w['x'],w['xprime']] == record['pairs'][w['pair_index']]
            assert ie.nonzero(cache[w['pair_index']][w['parameter']]);active_count+=1
        m=record['minor']
        if m['lower']:
            assert len(m['rows'])==len(m['columns'])==m['lower']
            d=ie.determinant([[cache[r][c] for c in m['columns']] for r in m['rows']])
            assert d is not None and ie.nonzero(d);minor_count+=1
        assert not record['old_not_recovered'];checked+=1
    return {'passed':True,'slots':checked,'active_witnesses':active_count,'nonzero_minors':minor_count,
            'replay_precision_dps':65,'manifest_files':len(manifest)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=DEFAULT)
    parser.add_argument('--audit',type=Path);args=parser.parse_args()
    if args.audit: print(json.dumps(audit(args.audit),indent=2))
    else: run(args.source)
