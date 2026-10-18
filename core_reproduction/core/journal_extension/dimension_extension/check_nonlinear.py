"""Independent pinned-source numerical check of the discovered closed form."""
import json
import argparse
from pathlib import Path
import sys
import numpy as np
from experiment import save,sha
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source_core'))
from source_host import inventory,construct,verify_sources
from adapter import source_states,source_features,encode
from engines import features

def run(output=None,proof=None):
    verify_sources();entry=next(e for e in inventory() if e['module']=='hubregtsen_encoding_circuit')
    obj=construct(entry,4,1)[0]
    info=json.loads((ROOT/'frozen_inputs/hubregtsen_encoding_circuit_n4_l1_local_z.json').read_text())
    rng=np.random.RandomState(20261004);errors=dict(features=0.,distance=0.,kernel=0.,source_gradient=0.)
    for i in range(1000):
        X=rng.uniform(-.99,.99,(2,2));theta=rng.uniform(-2,2,8)
        f=source_features(source_states(obj,X,theta),info['observables'])
        closed=-np.cos(X[:,np.arange(4)%2])*np.sin(theta[:4])
        beta=np.array([np.sum(np.sin(theta[[0,2]])**2),np.sum(np.sin(theta[[1,3]])**2)])
        c=np.cos(X[0])-np.cos(X[1]);d=float(beta@(c*c));native=float(np.sum((f[0]-f[1])**2))
        errors['features']=max(errors['features'],float(np.max(abs(f-closed))))
        errors['distance']=max(errors['distance'],abs(d-native))
        errors['kernel']=max(errors['kernel'],abs(np.exp(-.5*d)-np.exp(-.5*native)))
        if i<100:
            nf,j,_=features(info,encode(info['data_expressions'],X),theta,list(range(8)))
            native_grad=2*np.sum((nf[0]-nf[1])*(j[:,0]-j[:,1]),axis=1)
            closed_grad=np.zeros(8);closed_grad[:4]=np.sin(2*theta[:4])*c[np.arange(4)%2]**2
            errors['source_gradient']=max(errors['source_gradient'],float(np.max(abs(native_grad-closed_grad))))
    result=dict(passed=max(errors.values())<1e-10,input_pairs=1000,gradient_points=100,max_errors=errors,
                seed=20261004,script_sha256=sha(Path(__file__)),exact_proof_sha256=sha(proof or ROOT/'nonlinear_hubregtsen.json'),
                scope='Numerical consistency check; exact feature coefficients and analytic minor supply the proof')
    save(output or ROOT/'nonlinear_source_check.json',result);print(json.dumps(result,indent=2));assert result['passed']

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--proof',type=Path);args=parser.parse_args();run(args.output,args.proof)
