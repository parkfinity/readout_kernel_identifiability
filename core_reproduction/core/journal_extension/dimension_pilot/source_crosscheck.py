"""Numerical source/IR/Fourier consistency diagnostic, not identity acceptance."""
import argparse
import cmath
import hashlib
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import numpy as np
from mpmath import iv
import exact_engine as ex
import interval_engine as ie

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'cross_family_resource_v1/core'))
from source_host import inventory,construct,verify_sources
from adapter import source_states,source_features


def run(folder):
    verify_sources();entries={v['module']:v for v in inventory()};objects={}
    metadata=json.loads((folder/'run.json').read_text());errors=dict(features=0.,distance=0.,gradient_fd=0.)
    n_probes=0;n_coords=0
    for slot in metadata['scope']:
        key=(slot['module'],slot['n'],slot['layers'])
        if key not in objects: objects[key]=construct(entries[key[0]],key[1],key[2])[0]
        obj=objects[key];info=json.loads((folder/'inputs'/(slot['id']+'.json')).read_text())
        record=json.loads((folder/(slot['id']+'.json')).read_text())
        bases,d=ex.distance(info);theta=np.array([float(F(v)) for v in record['theta']]);iv.dps=50
        for index in (0,1):
            pair=record['pairs'][index];X=np.array([[float(F(v)) for v in x] for x in pair])
            f=source_features(source_states(obj,X,theta),info['observables'])
            fi=np.array([[ie.mid(v) for v in ie.features(info,record['theta'],x)[0]] for x in pair])
            errors['features']=max(errors['features'],float(np.max(abs(fi-f))))
            distance=float(np.sum(np.array(info['weights'])*(f[0]-f[1])**2))
            encoded=[]
            for x in pair: encoded += [ie.mid(ie.expression(b,[ie.rational(v) for v in x])) for b in bases]
            formula=0j
            for phase,c in d.items():
                angle=sum(float(a)*(theta[t] if t>=0 else 1)*(encoded[z] if z>=0 else 1) for t,z,a in phase)
                formula+=complex(float(c[0]),float(c[1]))*cmath.exp(1j*angle)
            errors['distance']=max(errors['distance'],abs(formula-distance))
            grad=ie.derivative(info,record['theta'],*pair)[1]
            for k in range(info['count']):
                h=1e-6;tp=theta.copy();tm=theta.copy();tp[k]+=h;tm[k]-=h
                fp=source_features(source_states(obj,X,tp),info['observables'])
                fm=source_features(source_states(obj,X,tm),info['observables'])
                fd=(np.sum(np.array(info['weights'])*(fp[0]-fp[1])**2)-np.sum(np.array(info['weights'])*(fm[0]-fm[1])**2))/(2*h)
                errors['gradient_fd']=max(errors['gradient_fd'],abs(fd-ie.mid(grad[k])));n_coords+=1
            n_probes+=1
    limits=dict(features=1e-10,distance=1e-10,gradient_fd=1e-6)
    return dict(passed=all(errors[k]<=limits[k] for k in errors),max_errors=errors,limits=limits,
                source_pair_probes=n_probes,source_fd_coordinates=n_coords,
                scope='Pinned sQUlearn source host; numerical diagnostic, not complete source-compiler formal proof')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    result=run(a.folder)
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['run_manifest_sha256']=hashlib.sha256((a.folder/'manifest.json').read_bytes()).hexdigest()
    if a.output: a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['passed']
