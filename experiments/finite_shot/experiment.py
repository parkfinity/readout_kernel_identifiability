"""Evaluate frozen designs. No metric selection or SVD direction choice on server."""
import argparse
from datetime import datetime,timezone
import platform
import time
import sympy
from core import *

def empirical(st,rec,spec,panel,point,shots,reps):
    th=np.array(rec['theta']);phi=np.array(rec['phi']);Q=np.array(rec['Q'])
    P=len(phi)//2;bases=spec['bases'];records=[];saved={}
    rootcode=['yz','kyri','hub_z'].index(spec['reference'])
    for direction in range(Q.shape[1]):
        v=Q[:,direction];pp=dense_states(st,th+.05*v,phi);pm=dense_states(st,th-.05*v,phi)
        fp=dense_features(st,th+.05*v,phi);fm=dense_features(st,th-.05*v,phi)
        ap=fp[:,:P]-fp[:,P:];am=fm[:,:P]-fm[:,P:]
        cp=outcome_covariance(pp,st.obs,bases);cm=outcome_covariance(pm,st.obs,bases)
        rng=np.random.default_rng(np.random.SeedSequence([2026102001,panel,rootcode,point,direction,shots]))
        hp,used=estimates(pp,st.obs,bases,shots,reps,rng)
        hm,_=estimates(pm,st.obs,bases,shots,reps,rng)
        for arm,key in spec['arms'].items():
            M=np.array(spec['metrics'][key]);vp,_=distance_variance(ap,cp,M,shots,bases)
            vm,_=distance_variance(am,cm,M,shots,bases);var=vp+vm
            signal_pairs=np.einsum('op,oq,qp->p',ap,M,ap)-np.einsum('op,oq,qp->p',am,M,am)
            if np.linalg.norm(signal_pairs)<1e-12:
                contrast=np.zeros(P);contrast[0]=1.;null=True
            else:
                contrast=signal_pairs/np.maximum(var,1e-24)
                contrast/=np.linalg.norm(contrast);null=False
            signal=float(signal_pairs@contrast);pred_var=float(var@(contrast**2))
            samples=(unbiased_dist(hp,M)-unbiased_dist(hm,M))@contrast
            sd=float(samples.std(ddof=1));mean=float(samples.mean());pred_sd=np.sqrt(pred_var)
            records.append(dict(direction=direction,arm=arm,candidate=key,requested_shots=shots,
                used_shots_per_input_state=used,total_shots_per_direction_replicate=used*4*P,
                replicates=reps,ideal_signal=signal,predicted_sd=pred_sd,empirical_mean=mean,
                empirical_sd=sd,predicted_snr=abs(signal)/pred_sd if pred_sd else None,
                empirical_snr=abs(signal)/sd if sd else None,
                variance_ratio=sd**2/pred_var if pred_var else None,
                bias_mc_se=(mean-signal)/(sd/np.sqrt(reps)) if sd else None,
                null_signal=null,contrast=contrast))
            saved['d%d_%s_s%d'%(direction,arm,shots)]=samples
    return records,saved

def run():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['pilot','full'],default='pilot');args=ap.parse_args()
    plan=load_plan();start=time.perf_counter()
    out=HERE/'results'/(args.mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    save(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,sympy=sympy.__version__))
    source=[p for p in HERE.rglob('*') if p.is_file() and 'results' not in p.parts and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.txt','.sh')]
    save(out/'provenance.json',{str(p.relative_to(HERE)):sha(p) for p in source})
    panels=plan['panels'][:1] if args.mode=='pilot' else plan['panels']
    budgets=[1024,4096] if args.mode=='pilot' else plan['full_shots']
    reps=64 if args.mode=='pilot' else plan['full_replicates']
    summaries=[]
    for panel in panels:
        for spec in panel['cases']:
            st=setting(spec['case']);points=panel['groups'][spec['reference']]['test']
            if args.mode=='pilot':points=points[:1]
            for i,rec in enumerate(points):
                t0=time.perf_counter();d=point_data(st,rec);cov=outcome_covariance(d['psi'],st.obs,spec['bases'])
                ana=[];mc=[];samples={}
                for shots in budgets:
                    for key,mat in spec['metrics'].items():
                        stats=sensitivity(d,np.array(mat),shots,spec['bases'],cov)
                        ana.append(dict(candidate=key,shots=shots,**stats))
                    records,draws=empirical(st,rec,spec,panel['panel'],i,shots,reps)
                    mc+=records;samples.update(draws)
                name='panel%d_%s_point%d'%(panel['panel'],spec['case'],i)
                record=dict(panel=panel['panel'],case=spec['case'],test_point=i,arms=spec['arms'],
                    bases=spec['bases'],theta=rec['theta'],target_dimension=d['Q'].shape[1],
                    initially_lost_dimension=d['L'].shape[1],analytic=ana,monte_carlo=mc,
                    elapsed_seconds=time.perf_counter()-t0)
                save(out/(name+'.json'),record)
                np.savez_compressed(out/(name+'_samples.npz'),**samples)
                comparisons=[]
                for shots in budgets:
                    get=lambda arm:next(a for a in ana if a['candidate']==spec['arms'][arm] and a['shots']==shots)
                    new=get('shot_selected')
                    for arm in ['fixed_diagonal','rank_selected','random_dense']:
                        base=get(arm)
                        comparisons.append(dict(shots=shots,baseline=arm,
                            new_rank=new['target_rank'],baseline_rank=base['target_rank'],
                            target_dimension=new['target_dimension'],
                            retained_sensitivity_ratio=new['retained_sensitivity']/base['retained_sensitivity'] if base['retained_sensitivity']>1e-12 else None,
                            new_weakest=new['weakest_sensitivity'],baseline_weakest=base['weakest_sensitivity']))
                summaries.append(dict(file=name+'.json',panel=panel['panel'],case=spec['case'],test_point=i,
                    comparisons=comparisons,elapsed_seconds=record['elapsed_seconds']))
                print(name,'seconds',round(record['elapsed_seconds'],2),'targets',d['Q'].shape[1],flush=True)
    save(out/'summary.json',dict(mode=args.mode,complete=True,plan_sha256=sha(HERE/'plan.json'),
        records=summaries,panels=len(panels),budgets=budgets,replicates=reps,
        elapsed_seconds=time.perf_counter()-start,
        scope='Held-out numerical sensitivity and ideal joint-shot sampling. No new exact certificates, hardware or classification results.'))
    print(json.dumps(dict(output=str(out),complete=True,records=len(summaries),elapsed_seconds=time.perf_counter()-start),indent=2))

if __name__=='__main__':run()
