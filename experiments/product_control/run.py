"""Same-outcome classical lift, exact covariance and paired Monte Carlo."""
from pathlib import Path
import sys, json, hashlib, platform, time
import numpy as np

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / 'readout_metric_shot_design_v2'
sys.path.insert(0, str(V2))
from core import (load_plan, setting, dense_states, dense_features,
                  basis_probabilities, estimates, unbiased_dist,
                  outcome_covariance, distance_variance, save, sha)

def subsets(obs):
    return [tuple(i for i, c in enumerate(w) if c == 'Z') for w in obs]

def lifted(halves, sets):
    return [np.stack([np.prod(h[:, list(s), :], axis=1) for s in sets], axis=1)
            for h in halves]

def product_cov(mu, sets, half):
    """mu: four by input; covariance of monomials of independent means."""
    means = np.stack([np.prod(mu[list(s)], axis=0) for s in sets], axis=1)
    second = mu**2 + (1-mu**2)/half
    cov = np.empty((mu.shape[1], len(sets), len(sets)))
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            shared, odd = sorted(set(a)&set(b)), sorted(set(a)^set(b))
            val = np.prod(second[shared], axis=0)*np.prod(mu[odd], axis=0)
            cov[:, i, j] = val-means[:, i]*means[:, j]
    return cov

def variance(delta, covmean, M):
    P = delta.shape[1]
    S = covmean[:P]+covmean[P:]
    a = M@delta
    return 2*np.einsum('op,poq,qp->p', a,S,a)+np.einsum('ij,pjk,kl,pli->p',M,S,M,S)

def main():
    start=time.perf_counter();plan=load_plan();st=setting('hub_zz');sets=subsets(st.obs)
    assert sets[:4] == [(0,), (1,), (2,), (3,)]
    server=HERE.parent/'readout_metric_shot_v2_server_audit/results/full_20261005T050625_181530Z'
    records=[];draws={};errors=dict(factorized_probability=0.,ideal_features=0.,direct_replay=0.)
    for panel in plan['panels']:
        pn=panel['panel'];spec=next(c for c in panel['cases'] if c['case']=='hub_zz')
        for point,rec in enumerate(panel['groups']['hub_z']['test']):
            theta=np.array(rec['theta']);phi=np.array(rec['phi']);Q=np.array(rec['Q']);P=len(phi)//2
            with np.load(server/('panel%d_hub_zz_point%d_samples.npz'%(pn,point))) as saved:
                for direction in range(Q.shape[1]):
                    f=[];psi=[];cov=[]
                    for sign in [1,-1]:
                        th=theta+sign*.05*Q[:,direction]
                        p=dense_states(st,th,phi);ff=dense_features(st,th,phi)
                        ints=np.arange(p.shape[1]);pr=np.ones_like(abs(p)**2)
                        for q in range(4):
                            pr*=(1+ff[q,:,None]*(1-2*((ints>>q)&1)))/2
                        errors['factorized_probability']=max(errors['factorized_probability'],float(np.max(abs(pr-basis_probabilities(p,'ZZZZ')))))
                        target=np.stack([np.prod(ff[list(s)],axis=0) for s in sets])
                        errors['ideal_features']=max(errors['ideal_features'],float(np.max(abs(target-ff))))
                        f.append(ff);psi.append(p);cov.append(outcome_covariance(p,st.obs,spec['bases']))
                    delta=[ff[:,:P]-ff[:,P:] for ff in f]
                    for shots in plan['full_shots']:
                        rng=np.random.default_rng(np.random.SeedSequence([2026102001,pn,2,point,direction,shots]))
                        hh=[estimates(p,st.obs,spec['bases'],shots,256,rng)[0] for p in psi]
                        ll=[lifted(h,sets) for h in hh]
                        pc=[product_cov(ff[:4],sets,shots//2) for ff in f]
                        for arm,key in spec['arms'].items():
                            M=np.array(spec['metrics'][key])
                            vd=sum(distance_variance(d,c,M,shots,spec['bases'])[0] for d,c in zip(delta,cov))
                            vl=sum(variance(d,c,M) for d,c in zip(delta,pc))
                            signalpairs=(np.einsum('op,oq,qp->p',delta[0],M,delta[0])-np.einsum('op,oq,qp->p',delta[1],M,delta[1]))
                            w=signalpairs/np.maximum(vd,1e-24)
                            if np.linalg.norm(signalpairs)<1e-12:w=np.eye(P)[0]
                            else:w/=np.linalg.norm(w)
                            signal=float(signalpairs@w)
                            direct=(unbiased_dist(hh[0],M)-unbiased_dist(hh[1],M))@w
                            lift=(unbiased_dist(ll[0],M)-unbiased_dist(ll[1],M))@w
                            old=saved['d%d_%s_s%d'%(direction,arm,shots)]
                            errors['direct_replay']=max(errors['direct_replay'],float(np.max(abs(old-direct))))
                            pv=float(vl@(w*w));dv=float(vd@(w*w))
                            assert pv>0 and pv<=dv*(1+1e-9)
                            name='p%d_t%d_d%d_%s_s%d'%(pn,point,direction,arm,shots)
                            draws[name+'_direct']=direct;draws[name+'_product']=lift
                            records.append(dict(panel=pn,point=point,direction=direction,arm=arm,shots=shots,
                                ideal_signal=signal,exact_direct_variance=dv,exact_product_variance=pv,
                                exact_variance_ratio=pv/dv,direct_mc_variance=float(direct.var(ddof=1)),
                                product_mc_variance=float(lift.var(ddof=1)),product_calibration=float(lift.var(ddof=1)/pv),
                                product_mean=float(lift.mean()),product_bias_mc_se=float((lift.mean()-signal)/(lift.std(ddof=1)/16)),
                                direct_mse=float(np.mean((direct-signal)**2)),product_mse=float(np.mean((lift-signal)**2))))
            print('panel',pn,'point',point,'complete',flush=True)
    summaries=[]
    for shots in plan['full_shots']:
        for pn in range(3):
            rr=[r for r in records if r['shots']==shots and r['panel']==pn and r['arm']=='shot_selected']
            summaries.append(dict(panel=pn,shots=shots,cells=len(rr),
                variance_ratio_median=float(np.median([r['exact_variance_ratio'] for r in rr])),
                variance_ratio_range=[float(min(r['exact_variance_ratio'] for r in rr)),float(max(r['exact_variance_ratio'] for r in rr))],
                empirical_mse_ratio_median=float(np.median([r['product_mse']/r['direct_mse'] for r in rr]))))
    assert max(errors.values())<1e-10
    out=HERE/'results';out.mkdir(exist_ok=True)
    save(out/'records.json',records);np.savez_compressed(out/'samples.npz',**draws)
    summary=dict(passed=True,records=len(records),replicates=256,errors=errors,panels=summaries,
        product_variance_calibration_quantiles=np.quantile([r['product_calibration'] for r in records],[.05,.5,.95]),
        source_plan_sha256=sha(V2/'plan.json'),script_sha256=sha(Path(__file__)),protocol_sha256=sha(HERE/'PROTOCOL.txt'),
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
        elapsed_seconds=time.perf_counter()-start,
        scope='Additional local control; direct samples replay existing Linux v2; product arm not yet server-reproduced.')
    save(out/'summary.json',summary);print(json.dumps(summary,indent=2,default=lambda x:x.tolist()))

if __name__=='__main__':main()
