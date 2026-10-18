"""Fixed-shot metric design; numerical sensitivity, not exact rank certification."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from legacy_design import (fg, matrix, probes, catalogue, cover, compatible,
                           dense_states, dense_features, basis_probabilities,
                           estimates, unbiased_dist, rank_stat)

HERE=Path(__file__).resolve().parent
NAMES=dict(yz='yz_cx_encoding_circuit_n2_l1_local_xyz',
           kyri='kyriienko_nonlinear_encoding_circuit_n2_l1_local_xyz',
           hub_z='hubregtsen_encoding_circuit_n4_l1_local_z')
CASES=['yz','kyri','hub_z','hub_zz']

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def setting(case):
    root='hub_z' if case=='hub_zz' else case
    d=json.loads((HERE/'frozen_inputs'/(NAMES[root]+'.json')).read_text())
    if case=='hub_zz':
        d=copy.deepcopy(d)
        for i in range(4):
            for j in range(i+1,4):
                w=['I']*4;w[i]=w[j]='Z';d['observables'].append(''.join(w))
        d['weights']=[1]*len(d['observables'])
    return fg.Setting(d,case)

def rank(a): return rank_stat(a)['rank']

def canonical_span(A):
    """Basis for column space, invariant to nondegenerate SVD sign choices."""
    if not A.size: return np.zeros((A.shape[0],0))
    u,_,_=np.linalg.svd(A,full_matrices=False);r=rank(A)
    P=u[:,:r]@u[:,:r].T;cols=[]
    for j in range(P.shape[1]):
        v=P[:,j].copy()
        for _ in range(2):
            for q in cols:v-=q*np.dot(q,v)
        if np.linalg.norm(v)>1e-8:
            v/=np.linalg.norm(v)
            if v[np.argmax(abs(v))]<0:v=-v
            cols.append(v)
        if len(cols)==r:break
    if len(cols)!=r:raise ValueError('Unstable canonical tangent basis')
    return np.stack(cols,axis=1) if cols else np.zeros((A.shape[0],0))

def target_spaces(p):
    f=p['f'];df=p['df'];P=f.shape[1]//2
    F=(df-df[:,:,0,None]).transpose(0,2,1).reshape(-1,df.shape[1])
    Q=canonical_span(F.T)
    a=f[:,:P]-f[:,P:];b=df[:,:,:P]-df[:,:,P:]
    J=2*np.einsum('op,okp->pk',a,b)/len(f)
    _,_,V=np.linalg.svd(J,full_matrices=True)
    N=V[rank(J):].T
    L=canonical_span(N@canonical_span((F@N).T))
    return Q,L

def point_data(st,record):
    th=np.array(record['theta']);phi=np.array(record['phi'])
    f,df=fg.NumericEngine(st).simulate(th,phi)
    P=f.shape[1]//2
    psi=dense_states(st,th,phi)
    return dict(theta=th,phi=phi,f=f,df=df,delta=f[:,:P]-f[:,P:],
                ddelta=df[:,:,:P]-df[:,:,P:],psi=psi,Q=np.array(record['Q']),L=np.array(record['L']))

def outcome_covariance(psi,obs,bases):
    """Covariance of one basis outcome; cross-basis features sampled independently."""
    O=len(obs);cov=np.zeros((len(psi),O,O));used=set()
    ints=np.arange(psi.shape[1])
    for b in bases:
        ids=[i for i,w in enumerate(obs) if i not in used and compatible(w,b)]
        used.update(ids)
        if not ids:continue
        signs=np.array([1-2*(sum((ints>>q)&1 for q,a in enumerate(obs[i]) if a!='I')%2) for i in ids])
        pr=basis_probabilities(psi,b);mu=pr@signs.T
        second=np.einsum('ps,os,qs->poq',pr,signs,signs)
        block=second-mu[:,:,None]*mu[:,None,:]
        for a,i in enumerate(ids):
            for c,j in enumerate(ids):cov[:,i,j]=block[:,a,c]
    if len(used)!=O:raise ValueError('Uncovered observables')
    return cov

def distance_variance(delta,cov,M,shots,bases):
    half=shots//(2*len(bases))
    if half<1:raise ValueError('Insufficient shots')
    P=delta.shape[1];S=(cov[:P]+cov[P:])/half
    Ma=M@delta
    v=2*np.einsum('op,poq,qp->p',Ma,S,Ma)+np.einsum('ij,pjk,kl,pli->p',M,S,M,S)
    if not np.isfinite(v).all() or v.min() < -1e-10:raise ValueError('Invalid variance')
    return np.maximum(v,0),2*half*len(bases)

def sensitivity(data,M,shots,bases,cov):
    J=2*np.einsum('op,oq,qkp->pk',data['delta'],M,data['ddelta'])
    Q=data['Q'];L=data['L'];r=rank(J@Q);d=Q.shape[1]
    v,used=distance_variance(data['delta'],cov,M,shots,bases)
    # Difference of independent theta+ and theta- estimates at the local limit.
    W=.1*(J@Q)/np.sqrt(np.maximum(2*v,1e-24))[:,None]
    sw=np.linalg.svd(W,compute_uv=False);s=np.linalg.svd(J@Q,compute_uv=False)
    return dict(target_dimension=d,target_rank=r,target_rank_tolerances=rank_stat(J@Q)['sensitivity_ranks'],source_kernel_rank=rank(J),
                initially_lost_dimension=L.shape[1],restored_subspace_rank=rank(J@L),
                singular_values=s,whitened_singular_values=sw,
                retained_sensitivity=float(sw[r-1]) if r else 0.,
                weakest_sensitivity=float(sw[d-1]) if d and r==d else 0.,
                rank_only_retained=float(s[r-1]) if r else 0.,used_shots=used)

def selector(scores,noise):
    key='retained_sensitivity' if noise else 'rank_only_retained'
    return max(scores,key=lambda name:(min(s['target_rank'] for s in scores[name]),
        float(np.mean([np.log(max(s[key],1e-30)) for s in scores[name]]))))

def load_plan(verify_package=True):
    package=HERE/'PACKAGE_MANIFEST.json'
    if verify_package and package.exists():
        for name,h in json.loads(package.read_text()).items():
            if sha(HERE/name)!=h:raise ValueError('Package file changed: '+name)
    manifest=json.loads((HERE/'FROZEN_MANIFEST.json').read_text())
    for name,h in manifest.items():
        if sha(HERE/name)!=h:raise ValueError('Frozen input changed: '+name)
    return json.loads((HERE/'plan.json').read_text())
