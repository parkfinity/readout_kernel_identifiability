"""Fixed-denominator, readout-aware resource survey. No wall-time speedup claim."""
import argparse, collections, copy, hashlib, inspect, itertools, json, platform, sys, time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'core'))
import qiskit
from qiskit import transpile
from source_host import construct, inventory, verify_sources
from adapter import adapt, encode, source_states, source_features, UnsupportedExpression
from pipeline import BASIS, boundary_transport, program, support
from certificates import invariant_generator, reduce_circuit
from structural_core import commute, conjugate_pauli
from engines import features, gram, kernel, alignment, angle_factors, objective_gradient


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def readout(n, kind):
    if kind == 'pauli_weight_le_2':
        obs = [''.join(p) for p in itertools.product('IXYZ', repeat=n)
               if 1 <= sum(c != 'I' for c in p) <= 2]
    else:
        obs = ['I' * q + a + 'I' * (n-q-1)
               for a in ('Z' if kind == 'local_z' else 'XYZ') for q in range(n)]
    return obs, np.ones(len(obs))


def gate_isometry(g, obs, w):
    if g.op == 'R':
        return invariant_generator(g.pauli, obs, w)[0]
    lookup = dict(zip(obs, w))
    return all(lookup.get(conjugate_pauli(o, g)) == v for o, v in zip(obs, w))


def isometry_tail(circuit, obs, w):
    """Elementary common-isometry suffix, generalized beyond local XYZ.

    Either the entire suffix is a common isometry, or all overlapping
    gates are common isometries confined to the candidate's support.
    The latter restriction prevents an entangler spreading the generator
    into a subsequent, initially disjoint data gate.
    This is a sufficient baseline, not complete symbolic simplification.
    """
    work = list(enumerate(circuit)); removed = []
    for i in range(len(work)-1, -1, -1):
        original, g = work[i]
        if g.op != 'R' or g.kind != 'param' or not gate_isometry(g, obs, w):
            continue
        suffix = [h for _,h in work[i+1:]]
        common = all(not h.data_dependent and gate_isometry(h, obs, w) for h in suffix)
        confined = all(not support(g).intersection(support(h)) or
                       (support(h).issubset(support(g)) and not h.data_dependent and
                        gate_isometry(h, obs, w)) for h in suffix)
        if common or confined:
            removed.append(original); work.pop(i)
    return tuple(g for _, g in work), sorted(removed)


def angle_tasks(p):
    tasks = []
    for k, g in enumerate(p):
        if g['op'] != 'R': continue
        terms = collections.defaultdict(float)
        for coef, ti, xi in g['terms']:
            if ti >= 0: terms[ti, xi] += coef
        if any(v != 0 for v in terms.values()): tasks.append(k)
    return tasks


def shared_features(info, E, theta):
    """Shift measured FEATURES per fused angle; chain rule through the RBF.

    This validates a simulator schedule, not finite-shot/QPU execution.
    Every symbol sharing an angle reuses its derivative. Shared symbols
    across gates accumulate all occurrence contributions.
    """
    f = features(info, E, theta)[0]
    jac = np.zeros((info['count'], len(E), len(info['observables'])))
    tasks = angle_tasks(info['program'])
    for k in tasks:
        g = info['program'][k]; shifted = []
        for sign in (1, -1):
            p = copy.deepcopy(info['program'])
            p[k]['terms'].append([sign*np.pi/2, -1, -1])
            shifted.append(features(dict(info, program=p), E, theta)[0])
        df = .5*(shifted[0]-shifted[1])
        _, factors = angle_factors(g, E, theta)
        for ti, factor in factors.items():
            jac[ti] += df*factor[:, None]
    return f, jac, {'angle_derivatives': len(tasks), 'feature_sweeps': 1+2*len(tasks)}


def shared_gradient(info, E, y, theta):
    f,jac,counters = shared_features(info,E,theta)
    K = gram(f,info['weights']); score,S = alignment(K,y)
    sensitivity = -np.sum(((S+S.T)*K)[:,:,None]*(f[:,None]-f[None,:])*info['weights'],axis=1)
    return score,np.einsum('pno,no->p',jac,sensitivity),counters


def kernel_jacobian(f,jac,w):
    return -gram(f,w)[None]*np.sum(np.asarray(w)*(f[:,None]-f[None,:])[None]*
                                  (jac[:,:,None]-jac[:,None,:]),axis=-1)


def maxerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.max(abs(a-b))) if a.size else 0.


def make_info(c, n, count, expr, obs, w):
    p = program(c, fuse_adjacent=True)
    ids = sorted({i for g in p if g['op']=='R' for coef,i,j in g['terms'] if i>=0 and coef!=0})
    return {'n': n, 'count': count, 'program': p, 'ids': ids,
            'data_expressions': expr, 'observables': obs, 'weights': w.tolist()}


def metrics(info):
    return {'active_symbols': len(info['ids']), 'fused_operations': len(info['program']),
            'trainable_angle_tasks': len(angle_tasks(info['program'])),
            'feature_sweeps_per_gradient': 1+2*len(angle_tasks(info['program']))}


def validate(obj, infos, folder, seeds=(812, 1907)):
    maxima = collections.defaultdict(float); samples = []; arrays = {}
    for seed in seeds:
        rng = np.random.RandomState(seed); X = rng.uniform(-.75,.75,(5,2))
        raw = infos['compiler']; theta = rng.uniform(.2,.8,raw['count']); y = rng.normal(size=5)
        E = encode(raw['data_expressions'], X)
        native = source_features(source_states(obj,X,theta),raw['observables'])
        K = gram(native,raw['weights'])
        maxima['source_feature'] = max(maxima['source_feature'],maxerr(native,features(raw,E,theta)[0]))
        try:
            alignment(K,y); objective_defined=True
        except ValueError:
            objective_defined=False
        fd = []; kfd=[]
        for j in range(raw['count']):
            pp = theta.copy(); mm = theta.copy(); pp[j]+=1e-6; mm[j]-=1e-6
            kp,km=kernel(raw,E,pp),kernel(raw,E,mm)
            kfd.append((kp-km)/2e-6)
            if objective_defined:fd.append((alignment(kp,y)[0]-alignment(km,y)[0])/2e-6)
        gs = []
        for name, info in infos.items():
            f,jac,counters=shared_features(info,E,theta);ka=gram(f,info['weights'])
            af,aj,_=features(info,E,theta,list(range(info['count'])))
            maxima['feature_jacobian'] = max(maxima['feature_jacobian'],maxerr(jac,aj))
            kj=kernel_jacobian(f,jac,info['weights'])
            if raw['count']:
                maxima['kernel_vs_fd'] = max(maxima['kernel_vs_fd'],maxerr(kj,kfd))
            maxima['kernel'] = max(maxima['kernel'],maxerr(K,ka))
            arrays[str(seed)+'_'+name+'_K'] = ka
            arrays[str(seed)+'_'+name+'_kernel_jacobian'] = kj
            if objective_defined:
                score,g,counters = shared_gradient(info,E,y,theta)
                ga = objective_gradient(info,E,y,theta,'batched')[1]
                maxima['angle_vs_analytic'] = max(maxima['angle_vs_analytic'],maxerr(g,ga))
                maxima['angle_vs_fd'] = max(maxima['angle_vs_fd'],maxerr(g,fd))
                gs.append(g);arrays[str(seed)+'_'+name+'_gradient'] = g
        if objective_defined:
            maxima['gradient_preservation'] = max(maxima['gradient_preservation'],maxerr(gs[1],gs[2]))
        # Deterministic two-update regression: compare both shared-angle arms.
        ts = {m: theta.copy() for m in ('boundary','transport')}
        for step in range(2 if objective_defined else 0):
            for m in ts: ts[m] += .1*shared_gradient(infos[m],E,y,ts[m])[1]
            maxima['trajectory'] = max(maxima['trajectory'], maxerr(ts['boundary'],ts['transport']))
        pred = []
        for m in ts:
            kk = kernel(infos[m],E,ts[m]); pred.append(kk[3:,:3] @ np.linalg.solve(kk[:3,:3]+.1*np.eye(3),y[:3]))
            arrays[str(seed)+'_'+m+'_theta_final'] = ts[m]
            arrays[str(seed)+'_'+m+'_prediction'] = pred[-1]
        maxima['prediction'] = max(maxima['prediction'],maxerr(*pred))
        arrays[str(seed)+'_X'] = X; arrays[str(seed)+'_theta'] = theta; arrays[str(seed)+'_y'] = y
        samples.append({'seed':seed,'inputs':5,'parameter_count':raw['count'], 'finite_difference_coordinates':raw['count'],
                        'objective_defined':objective_defined,'training_updates':2 if objective_defined else 0})
    limits = {k:(2e-7 if k in ('angle_vs_fd','kernel_vs_fd') else 1e-9 if k in ('angle_vs_analytic','gradient_preservation','trajectory','prediction','feature_jacobian') else 1e-10) for k in maxima}
    np.savez_compressed(folder/'validation_arrays.npz',**arrays)
    return {'passed':all(maxima[k]<=limits[k] for k in maxima),'max_errors':dict(maxima),
            'limits':limits,'samples':samples,'array_file':'validation_arrays.npz',
            'scope':'Qiskit source features at base points; compiled-IR finite differences; ideal shared-angle simulator; two-update regression, no timing or QPU test'}


def run(out):
    out.mkdir(parents=True,exist_ok=False); verify_sources()
    scope = json.loads((ROOT/'frozen_scope.json').read_text()); entries = {e['module']:e for e in inventory()}
    config = {'scope_sha256':sha(ROOT/'frozen_scope.json'),'code_sha256':sha(Path(__file__)),
              'seed_policy':'constructor defaults (random circuit seed 0); probes 812,1907',
              'revision':'r2_explicit_undefined_alignment_handling',
              'compiler':{'optimization_level':3,'seed_transpiler':71,'basis':BASIS},
              'numpy':np.__version__,'qiskit':qiskit.__version__,'python':sys.version,'platform':platform.platform(),
              'excluded_modules':[m for m in scope['modules'] if m['excluded_reason']],
              'baseline':'compile, readout-aware common-isometry tail, restricted boundary, adjacent fusion',
              'no_performance_timing':True}
    save(out/'run.json',config)  # Freeze before outcomes.
    rows = []; cache = {}
    for slot in scope['slots']:
        row = {k:slot[k] for k in ('id','module','n','layers','readout','duplicate_of','family_group')}
        if slot['duplicate_of']:
            row['status']='DUPLICATE'; rows.append(row); continue
        folder = out/slot['id']; folder.mkdir(); start = time.perf_counter()
        key = (slot['module'],slot['n'],slot['layers'])
        try:
            if key not in cache:
                obj,qc,x,t,kwargs = construct(entries[slot['module']],slot['n'],slot['layers'])
                compiled = transpile(qc,optimization_level=3,basis_gates=BASIS,seed_transpiler=71,
                                     approximation_degree=1.,num_processes=1)
                c,expr,meta = adapt(compiled,x,t)
                cache[key] = obj,c,expr,meta,len(t),kwargs
            obj,c,expr,meta,count,kwargs = cache[key]
            obs,w = readout(slot['n'],slot['readout'])
            constructor = {'passed_kwargs':kwargs, 'signature':str(inspect.signature(type(obj).__init__)),
                           'get_params':repr(obj.get_params())}
        except Exception as error:
            row.update(status='UNSUPPORTED_CONSTRUCTION_OR_ADAPTER',reason=repr(error))
            rows.append(row); save(folder/'result.json',row); print(row['id'],row['status'],flush=True); continue
        tail,removed = isometry_tail(c,obs,w)
        provenance = [p for i,p in enumerate(meta['gate_provenance']) if i not in removed]
        tick = time.perf_counter(); boundary,bp = boundary_transport(tail,obs,w,provenance)
        bt = time.perf_counter()-tick
        tick = time.perf_counter(); transport,tp = reduce_circuit(boundary,obs,w,'transport_metric')
        tt = time.perf_counter()-tick
        infos = {m:make_info(g,slot['n'],count,expr,obs,w)
                 for m,g in [('compiler',c),('boundary',boundary),('transport',transport)]}
        row.update(constructor=constructor,parameter_count=count,metrics={m:metrics(v) for m,v in infos.items()},
                   timing_diagnostic_seconds={'boundary':bt,'extra_transport':tt},
                   removed_occurrences=len(tp['certificates']),tail_removed_indices=removed)
        row['delta'] = {k:row['metrics']['boundary'][k]-row['metrics']['transport'][k] for k in metrics(infos['boundary'])}
        row['unknown_symbols'] = infos['transport']['ids']
        row['source_to_ir'] = meta
        row['boundary_proof'] = bp; row['transport_proof'] = tp
        save(folder/'programs.json',infos)
        try:
            row['validation'] = validate(obj,infos,folder)
            row['status'] = ('PASS_TRAINABLE' if count else 'PASS_PARAMETER_FREE') if row['validation']['passed'] else 'VALIDATION_FAILED'
            if row['validation']['passed'] and not all(s['objective_defined'] for s in row['validation']['samples']):
                row['status']='PASS_KERNEL_ONLY_ALIGNMENT_UNDEFINED'
        except Exception as error:
            row.update(status='VALIDATION_FAILED',reason=repr(error))
        row['diagnostic_elapsed_seconds'] = time.perf_counter()-start
        rows.append(row); save(folder/'result.json',row)
        print(row['id'],row['status'],row['delta'],flush=True)
    counters = dict(collections.Counter(r['status'] for r in rows))
    positive = [r['id'] for r in rows if r['status'].startswith('PASS') and
                (r['delta']['fused_operations']>0 or r['delta']['trainable_angle_tasks']>0)]
    summary = {'passed':not any(r['status']=='VALIDATION_FAILED' for r in rows),'nominal_slots':len(rows),
               'statuses':counters,'execution_resource_positive_slots':positive,
               'additional_symbol_slots':[r['id'] for r in rows if r['status'].startswith('PASS') and r['delta']['active_symbols']>0],
               'rows':rows,'interpretation':'Fixed source/default-constructor census, not prevalence or independent family count. No wall-time or QPU measurements.'}
    save(out/'summary.json',summary)
    save(out/'manifest.json',{'files':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]})
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
    return summary['passed']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    out=args.output or ROOT/'results'/('survey_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    sys.exit(0 if run(out) else 1)
