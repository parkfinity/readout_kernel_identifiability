"""Run every predeclared setting and preserve all outcomes and fixed designs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
import sympy
from design import (fg,probes,jac,rank_stat,score,catalogue,words,cover,measured,
                    greedy_bases,dense_states,dense_features,estimates,unbiased_dist)

ROOT=Path(__file__).resolve().parent

def save(path,data):
    path.write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')

def evaluate(points,M,idx=None):
    return [rank_stat(jac(p,M,idx)) for p in points]

def run():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    out=args.output or ROOT/'results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True,exist_ok=False)
    sources=sorted(p for p in (ROOT/'frozen_inputs').glob('*.json') if p.name!='scope.json')
    assert len(sources)==30
    provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in ROOT.rglob('*') if p.is_file() and (p.suffix in ('.py','.json','.txt','.sh'))
                and 'results' not in p.parts and '__pycache__' not in p.parts}
    save(out/'provenance.json',provenance)
    save(out/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,sympy=sympy.__version__,platform=platform.platform()))
    metric_rows=[]; cache={}; cross=[]; start=time.perf_counter()
    for ordinal,path in enumerate(sources):
        t0=time.perf_counter(); info=json.loads(path.read_text()); st=fg.Setting(info,path.stem)
        assert st.fixed_frequency and st.exact_supported
        design=probes(st,2026100401,3); val=probes(st,2026100402,5)
        mats=catalogue(len(st.obs),2026100403)
        diag=[k for k in mats if k in ('identity','fixed_diagonal') or 'random_diagonal' in k]
        bestdiag=max(diag,key=lambda k:score(design,mats[k]))
        bestdense=max(mats,key=lambda k:score(design,mats[k]))
        arms={'identity':'identity','fixed_diagonal':'fixed_diagonal',
              'random_diagonal':'random_diagonal_00','random_dense':'random_dense_00',
              'selected_diagonal':bestdiag,'selected_dense':bestdense}
        results={arm:dict(candidate=key,metric=mats[key],validation=evaluate(val,mats[key])) for arm,key in arms.items()}
        random_panel={key:[s['rank'] for s in evaluate(val,M)] for key,M in mats.items() if key.startswith('random')}
        feature_ranks=[rank_stat((p['df']-p['df'][:,:,0,None]).transpose(0,2,1).reshape(-1,st.K)) for p in val]
        # Independent dense state-vector features and central differences, all settings.
        p=val[0]; phi=p['phi'][:4]; th=p['theta']; h=1e-5
        feat=dense_features(st,th,phi)
        fd=np.stack([(dense_features(st,th+np.eye(st.K)[k]*h,phi)-dense_features(st,th-np.eye(st.K)[k]*h,phi))/(2*h) for k in range(st.K)],axis=1)
        check=dict(setting=path.stem,feature_error=float(np.max(abs(feat-p['f'][:,:4]))),gradient_error=float(np.max(abs(fd-p['df'][:,:,:4]))))
        assert check['feature_error']<1e-11 and check['gradient_error']<2e-8
        cross.append(check)
        row=dict(setting=path.stem,n=st.n,K=st.K,observables=st.obs,
                 basis_schedule=cover(st.obs),centered_feature_rank_diagnostics=feature_ranks,
                 arms=results,random_panel_ranks=random_panel,
                 design_theta=[p['theta'] for p in design],validation_theta=[p['theta'] for p in val],
                 elapsed_seconds=time.perf_counter()-t0)
        metric_rows.append(row); cache[path.stem]=(st,results)
        save(out/(path.stem+'.json'),row)
        ranks={k:[s['rank'] for s in v['validation']] for k,v in results.items()}
        print(path.stem,ranks,flush=True)
    save(out/'independent_checks.json',cross)

    readout_rows=[]
    for path in [p for p in sources if p.stem.endswith('_local_xyz')]:
        t0=time.perf_counter(); info=json.loads(path.read_text()); info['observables']=words(info['n']); info['weights']=[1]*len(info['observables'])
        st=fg.Setting(info,path.stem); design=probes(st,2026100411,3); val=probes(st,2026100412,5)
        greedy,neval=greedy_bases(design,st.obs)
        fixed=['Z'*st.n,'X'*st.n,'Y'*st.n]
        import itertools
        bank=[''.join(x) for x in itertools.product('XYZ',repeat=st.n)]
        rng=np.random.default_rng(2026100413)
        schedules={'standard':fixed,'rank_greedy':greedy}
        schedules.update({'random_%02d'%i:rng.choice(bank,3,replace=False).tolist() for i in range(16)})
        records=[]
        for method,bases in schedules.items():
            for B in (1,2,3):
                idx=measured(st.obs,bases[:B]); w=np.linspace(1,4,len(st.obs))[idx]; M=np.diag(w/w.sum())
                records.append(dict(method=method,budget=B,bases=bases[:B],observables=[st.obs[i] for i in idx],validation=evaluate(val,M,idx)))
        row=dict(circuit=path.stem.replace('_local_xyz',''),n=st.n,K=st.K,
                 greedy_candidate_evaluations=neval,records=records,elapsed_seconds=time.perf_counter()-t0)
        readout_rows.append(row)
        save(out/('readout_'+row['circuit']+'.json'),row)
        print('READOUT',row['circuit'],'greedy',greedy,flush=True)

    shot_rows=[]
    names=['yz_cx_encoding_circuit_n2_l1_local_xyz',
           'kyriienko_nonlinear_encoding_circuit_n2_l1_local_xyz',
           'hubregtsen_encoding_circuit_n4_l1_local_z']
    for name in names:
        st,arms=cache[name]; p=probes(st,2026100421,1,pairs=16)[0]
        M0=arms['identity']['metric']; M1=arms['selected_dense']['metric']
        J0=jac(p,M0); J1=jac(p,M1); _,s,V=np.linalg.svd(J0,full_matrices=True)
        r=rank_stat(J0)['rank']; N=V[r:].T
        if N.shape[1]==0 or np.linalg.norm(J1@N)<1e-9:
            shot_rows.append(dict(setting=name,status='no detectable restored direction at held-out point')); continue
        _,_,w=np.linalg.svd(J1@N,full_matrices=False); v=N@w[0]; v/=np.linalg.norm(v)
        thp=p['theta']+.05*v; thm=p['theta']-.05*v
        fp=dense_features(st,thp,p['phi']); fm=dense_features(st,thm,p['phi'])
        ap=fp[:,:16]-fp[:,16:]; am=fm[:,:16]-fm[:,16:]
        truth={arm:np.einsum('op,oq,qp->p',ap,M,ap)-np.einsum('op,oq,qp->p',am,M,am) for arm,M in [('identity',M0),('selected_dense',M1)]}
        # A fixed linear contrast matched to the ideal selected-metric difference.
        # Both methods and every shot budget use the SAME contrast and direction.
        contrast=truth['selected_dense']/np.linalg.norm(truth['selected_dense'])
        psip=dense_states(st,thp,p['phi']); psim=dense_states(st,thm,p['phi'])
        bases=cover(st.obs)
        for shots in (256,1024,4096):
            rng=np.random.default_rng(2026100422+shots)
            hp,used=estimates(psip,st.obs,bases,shots,128,rng)
            hm,_=estimates(psim,st.obs,bases,shots,128,rng)
            for arm,M in [('identity',M0),('selected_dense',M1)]:
                samples=(unbiased_dist(hp,M)-unbiased_dist(hm,M))@contrast
                signal=float(truth[arm]@contrast); sd=float(np.std(samples,ddof=1)); mean=float(np.mean(samples))
                shot_rows.append(dict(setting=name,method=arm,requested_shots_per_input_state=shots,
                         used_shots_per_input_state=used,measurement_bases=bases,
                         total_shots_per_replicate=used*64,replicates=128,
                         signal=signal,mean=mean,sd=sd,bias=mean-signal,mean_mc_se=sd/np.sqrt(128),
                         snr=abs(signal)/sd,positive_fraction=float(np.mean(samples>0)),
                         theta=p['theta'],direction=v,perturbation_total=.1,
                         contrast=contrast,scope='ideal joint-shot simulation; locally chosen direction; not a hypothesis-test power or hardware result'))
        print('SHOTS',name,flush=True)
    save(out/'finite_shots.json',shot_rows)
    save(out/'metric_summary.json',metric_rows)
    save(out/'readout_summary.json',readout_rows)
    summary=dict(settings=len(metric_rows),circuits=len(readout_rows),
                 design_points=3,validation_points=5,metric_candidates=34,
                 elapsed_seconds=time.perf_counter()-start,
                 max_feature_error=max(c['feature_error'] for c in cross),
                 max_gradient_error=max(c['gradient_error'] for c in cross),
                 rank_scope='numerical local rank diagnostics; exact controls are separate',
                 classification_accuracy_evaluated=False,hardware_executed=False)
    save(out/'summary.json',summary)
    print(json.dumps(dict(output=str(out),**summary),indent=2),flush=True)

if __name__=='__main__': run()
