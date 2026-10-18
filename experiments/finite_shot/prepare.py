"""Maintainer-only design preparation. Servers consume frozen plan.json."""
import time
import numpy as np
from core import *

def main():
    if (HERE/'plan.json').exists():raise SystemExit('Refuse to overwrite frozen plan; use a new version.')
    start=time.perf_counter();allpanels=[]
    for panel in range(3):
        groups={}
        for idx,key in enumerate(['yz','kyri','hub_z']):
            st=setting(key);groups[key]={}
            for split,count,offset in [('design',3,0),('test',5,1)]:
                points=probes(st,2026100500+100*panel+10*idx+offset,count,16)
                rec=[]
                for p in points:
                    Q,L=target_spaces(p)
                    rec.append(dict(theta=p['theta'],x=p['x'],phi=p['phi'],Q=Q,L=L))
                groups[key][split]=rec
        cases=[]
        for idx,key in enumerate(CASES):
            st=setting(key);root='hub_z' if key=='hub_zz' else key
            mats=catalogue(len(st.obs),2026101000+100*panel+idx)
            bases=cover(st.obs);scores={k:[] for k in mats}
            for p in groups[root]['design']:
                d=point_data(st,p);cov=outcome_covariance(d['psi'],st.obs,bases)
                for name,M in mats.items():scores[name].append(sensitivity(d,M,1024,bases,cov))
            arms=dict(identity='identity',fixed_diagonal='fixed_diagonal',
                      random_diagonal='random_diagonal_00',random_dense='random_dense_00',
                      rank_selected=selector(scores,False),shot_selected=selector(scores,True))
            cases.append(dict(case=key,reference=root,observables=st.obs,bases=bases,
                              metrics=mats,arms=arms,design_scores=scores))
            print('panel',panel,key,arms,flush=True)
        allpanels.append(dict(panel=panel,groups=groups,cases=cases))
    save(HERE/'plan.json',dict(version=2,panels=allpanels,selection_shots=1024,
        full_shots=[256,1024,4096,16384],full_replicates=256,perturbation=.1,
        preparation_seconds=time.perf_counter()-start))
    paths=[HERE/'plan.json',HERE/'PROTOCOL.txt']+sorted((HERE/'frozen_inputs').glob('*.json'))
    save(HERE/'FROZEN_MANIFEST.json',{str(p.relative_to(HERE)):sha(p) for p in paths})

if __name__=='__main__':main()
