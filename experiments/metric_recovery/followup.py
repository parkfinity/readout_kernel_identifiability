import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from design import *
from experiment import save,evaluate

def run():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',type=Path);args=ap.parse_args()
    run=args.pilot or sorted((HERE/'results').glob('*/readout_summary.json'))[-1].parent
    target=run/'strong_baseline_followup';target.mkdir(exist_ok=False)
    one=[]
    for row in json.loads((run/'readout_summary.json').read_text()):
        info=json.loads((HERE/'frozen_inputs'/(row['circuit']+'_local_xyz.json')).read_text())
        info['observables']=words(info['n']);info['weights']=[1]*len(info['observables'])
        st=fg.Setting(info);design=probes(st,2026100411,3);val=probes(st,2026100412,5)
        bank=[''.join(b) for b in itertools.product('XYZ',repeat=st.n)]
        def var_score(b):
            idx=measured(st.obs,[b])
            return float(np.mean([np.var(p['f'][idx],axis=1).mean() for p in design]))
        bvar=max(bank,key=var_score)
        bg=next(r['bases'][0] for r in row['records'] if r['method']=='rank_greedy' and r['budget']==1)
        methods={'all_X':'X'*st.n,'all_Y':'Y'*st.n,'all_Z':'Z'*st.n,'variance_selected':bvar,'rank_selected':bg,'rank_selected_identity':bg}
        records=[]
        for method,b in methods.items():
            idx=measured(st.obs,[b]);w=np.linspace(1,4,len(st.obs))[idx]
            if method=='rank_selected_identity':w=np.ones(len(idx))
            M=np.diag(w/w.sum())
            records.append(dict(method=method,basis=b,observables=[st.obs[i] for i in idx],validation=evaluate(val,M,idx)))
        one.append(dict(circuit=row['circuit'],K=st.K,records=records))
        print(row['circuit'],{v['method']:v['validation'][0]['rank'] for v in records},flush=True)
    save(target/'one_basis_baselines.json',one)

    old=json.loads((run/'finite_shots.json').read_text());rows=[]
    for name in sorted(set(r['setting'] for r in old if 'theta' in r)):
        base=next(r for r in old if r['setting']==name and 'theta' in r)
        metrics=json.loads((run/(name+'.json')).read_text())['arms']
        st=fg.Setting(json.loads((HERE/'frozen_inputs'/(name+'.json')).read_text()))
        p=probes(st,2026100421,1,16)[0];th=np.array(base['theta']);v=np.array(base['direction']);contrast=np.array(base['contrast'])
        assert np.allclose(th,p['theta'])
        psip=dense_states(st,th+.05*v,p['phi']);psim=dense_states(st,th-.05*v,p['phi'])
        fp=dense_features(st,th+.05*v,p['phi']);fm=dense_features(st,th-.05*v,p['phi'])
        dp=fp[:,:16]-fp[:,16:];dm=fm[:,:16]-fm[:,16:];bases=cover(st.obs)
        for shots in (256,1024,4096):
            rng=np.random.default_rng(2026100422+shots)
            hp,used=estimates(psip,st.obs,bases,shots,128,rng);hm,_=estimates(psim,st.obs,bases,shots,128,rng)
            for method,arm in metrics.items():
                M=np.array(arm['metric']);signal=float((np.einsum('op,oq,qp->p',dp,M,dp)-np.einsum('op,oq,qp->p',dm,M,dm))@contrast)
                samples=(unbiased_dist(hp,M)-unbiased_dist(hm,M))@contrast;sd=float(samples.std(ddof=1))
                rows.append(dict(setting=name,method=method,shots_requested=shots,shots_used=used,
                                 signal=signal,mean=float(samples.mean()),sd=sd,snr=abs(signal)/sd,
                                 total_shots_per_replicate=used*64,replicates=128,
                                 positive_fraction=float(np.mean(samples>0))))
    save(target/'finite_shot_baselines.json',rows)

if __name__=='__main__':run()
