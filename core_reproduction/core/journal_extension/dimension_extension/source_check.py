"""Pinned source and resource-arm numerical cross-check; not identity proof."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import numpy as np
from mpmath import iv
import interval_engine as ie
from experiment import save,sha
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source_core'))
from source_host import inventory,construct,verify_sources
from adapter import source_states,source_features,encode
from engines import features,gram

def run(folder):
    provenance=verify_sources();entries={v['module']:v for v in inventory()};objects={};records=[]
    metadata=json.loads((folder/'run.json').read_text());total_coordinates=0
    for slot in metadata['scope']:
        key=(slot['module'],slot['n'],slot['layers'])
        if key not in objects:objects[key]=construct(entries[key[0]],key[1],key[2])[0]
        obj=objects[key];info=json.loads((folder/'inputs'/(slot['id']+'.json')).read_text())
        r=json.loads((folder/(slot['id']+'.json')).read_text())
        errors=dict(source_features=0.,interval_gradient_fd=0.,arm_kernel=0.,arm_gradient=0.)
        theta=np.array([float(F(v)) for v in r['theta']]);iv.dps=50
        for j in (0,1):
            pair=r['pairs'][j];X=np.array([[float(F(v)) for v in x] for x in pair]);E=encode(info['data_expressions'],X)
            fs=source_features(source_states(obj,X,theta),info['observables'])
            fi=np.array([[ie.mid(v) for v in ie.features(info,r['theta'],x)[0]] for x in pair])
            errors['source_features']=max(errors['source_features'],float(np.max(abs(fs-fi))))
            grad=np.array([ie.mid(v) for v in ie.derivative(info,r['theta'],*pair)[1]])
            for k in range(info['count']):
                h=1e-6;tp=theta.copy();tm=theta.copy();tp[k]+=h;tm[k]-=h
                fp=source_features(source_states(obj,X,tp),info['observables']);fm=source_features(source_states(obj,X,tm),info['observables'])
                fd=(np.sum(np.array(info['weights'])*(fp[0]-fp[1])**2)-np.sum(np.array(info['weights'])*(fm[0]-fm[1])**2))/(2*h)
                errors['interval_gradient_fd']=max(errors['interval_gradient_fd'],abs(fd-grad[k]));total_coordinates+=1
        rng=np.random.RandomState(20261003)
        for j in range(20):
            X=rng.uniform(-.8,.8,(5,2));t=rng.uniform(-1,1,info['count']);E=encode(info['data_expressions'],X)
            fs=source_features(source_states(obj,X,t),info['observables']);K=gram(fs,info['weights'])
            f,df,_=features(info,E,t,list(range(info['count'])))
            w=np.array(info['weights']);dj=2*np.sum(w*(f[:,None]-f[None,:])[None]*(df[:,:,None]-df[:,None,:]),axis=-1)
            for arm in r['resources']['programs'].values():
                af,adf,_=features(arm,E,t,list(range(info['count'])))
                aj=2*np.sum(w*(af[:,None]-af[None,:])[None]*(adf[:,:,None]-adf[:,None,:]),axis=-1)
                errors['arm_kernel']=max(errors['arm_kernel'],float(np.max(abs(gram(af,w)-K))))
                errors['arm_gradient']=max(errors['arm_gradient'],float(np.max(abs(aj-dj))))
        limits=dict(source_features=1e-10,interval_gradient_fd=1e-6,arm_kernel=1e-10,arm_gradient=1e-9)
        records.append(dict(id=slot['id'],max_errors=errors,passed=all(errors[k]<limits[k] for k in errors)))
        print(slot['id'],records[-1]['passed'],flush=True)
    return dict(passed=all(r['passed'] for r in records),slots=len(records),source_pair_probes=2*len(records),
                source_fd_coordinates=total_coordinates,random_resource_probes=20*len(records),
                max_errors={k:max(r['max_errors'][k] for r in records) for k in records[0]['max_errors']},
                rows=records,source_commit=provenance['commit'],script_sha256=sha(Path(__file__)),
                run_manifest_sha256=sha(folder/'manifest.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.folder);save(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));assert result['passed']
