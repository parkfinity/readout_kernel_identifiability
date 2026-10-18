"""Readout/metric pilot. Numerical diagnostics are not rank certificates."""
import itertools
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
import fourier_gram as fg

PAULI = {'I': np.eye(2), 'X': np.array([[0,1],[1,0]]),
         'Y': np.array([[0,-1j],[1j,0]]), 'Z': np.diag([1,-1])}

def matrix(word):
    out = np.ones((1,1), complex)
    for p in word[::-1]:
        out = np.kron(out, PAULI[p])
    return out

def dense_states(st, theta, phi):
    """Independent dense matrix implementation, little-endian wire order."""
    psi = np.zeros((len(phi), st.N), complex); psi[:,0] = 1
    eye = np.eye(st.N)
    for g in st.gates:
        if g['op'] == 'R':
            angle = sum(float(c)*theta[k] for k,c in g['th'].items())
            angle = angle + sum(float(c)*phi[:,b] for b,c in g['da'].items())
            angle = np.broadcast_to(angle, (len(phi),))
            psi = np.cos(angle/2)[:,None]*psi - 1j*np.sin(angle/2)[:,None]*(psi @ matrix(g['pauli']).T)
        elif g['op'] == 'H':
            q = g['wires'][0]
            x = 'I'*q+'X'+'I'*(st.n-q-1); z = 'I'*q+'Z'+'I'*(st.n-q-1)
            psi = psi @ ((matrix(x)+matrix(z))/np.sqrt(2)).T
        else:
            c,t = g['wires']; perm = np.arange(st.N)
            perm = perm ^ (((perm >> c)&1) << t)
            psi = psi @ eye[perm].T
    return psi

def dense_features(st, theta, phi):
    psi = dense_states(st, theta, phi)
    return np.stack([np.real(np.sum(psi.conj()*(psi@matrix(o).T),axis=1)) for o in st.obs])

def probes(st, seed, ntheta, pairs=96):
    rng = np.random.default_rng(seed)
    ans=[]
    eng = fg.NumericEngine(st)
    for _ in range(ntheta):
        th = rng.uniform(-np.pi,np.pi,st.K)
        x = rng.uniform(-.95,.95,(2*pairs,2))
        phi = np.array([st.base_angles(v) for v in x])
        f,df = eng.simulate(th,phi)
        ans.append(dict(theta=th,x=x,phi=phi,f=f,df=df,
                        delta=f[:,:pairs]-f[:,pairs:],
                        ddelta=df[:,:,:pairs]-df[:,:,pairs:]))
    return ans

def jac(p, M, idx=None):
    a,b = p['delta'],p['ddelta']
    if idx is not None: a,b = a[idx],b[idx]
    return 2*np.einsum('op,oq,qkp->pk',a,M,b,optimize=True)/np.sqrt(a.shape[1])

def rank_stat(J, rtol=1e-9):
    s = np.linalg.svd(J,compute_uv=False)
    threshold = max(1e-11,rtol*(s[0] if len(s) else 0))
    r = int(np.sum(s>threshold))
    return dict(rank=r,singular_values=s.tolist(),smallest=float(s[r-1]) if r else 0,
                threshold=threshold, sensitivity_ranks=[int(np.sum(s>max(1e-11,t*(s[0] if len(s) else 0)))) for t in (1e-8,1e-9,1e-10)])

def score(points,M,idx=None):
    stats=[rank_stat(jac(p,M,idx)) for p in points]
    return min(s['rank'] for s in stats), float(np.mean([np.log(max(s['smallest'],1e-30)) for s in stats]))

def catalogue(O, seed):
    rng=np.random.default_rng(seed)
    mats={'identity':np.eye(O)/O, 'fixed_diagonal':np.diag(np.linspace(1,4,O))/np.linspace(1,4,O).sum()}
    for j in range(16):
        w=rng.uniform(1,4,O); mats['random_diagonal_%02d'%j]=np.diag(w/w.sum())
    for j in range(16):
        Q,_=np.linalg.qr(rng.normal(size=(O,O)))
        w=np.linspace(1,4,O); mats['random_dense_%02d'%j]=(Q*w[None,:])@Q.T/w.sum()
    return mats

def words(n):
    return [''.join(w) for w in itertools.product('IXYZ',repeat=n) if 1<=sum(a!='I' for a in w)<=2]

def compatible(word,basis): return all(a=='I' or a==b for a,b in zip(word,basis))

def measured(obs,bases):
    return [i for i,o in enumerate(obs) if any(compatible(o,b) for b in bases)]

def cover(obs):
    """Deterministic greedy QWC schedule. Counts are an upper bound, not optimum."""
    bases=[''.join(b) for b in itertools.product('XYZ',repeat=len(obs[0]))]
    remaining=set(range(len(obs))); out=[]
    while remaining:
        b=max(bases,key=lambda b:sum(compatible(obs[i],b) for i in remaining))
        out.append(b); remaining-={i for i in remaining if compatible(obs[i],b)}
    return out

def readout_score(points,obs,bases):
    idx=measured(obs,bases)
    w=np.linspace(1,4,len(obs))[idx]; M=np.diag(w/w.sum())
    return score(points,M,idx)

def greedy_bases(points,obs,budget=3):
    bank=[''.join(b) for b in itertools.product('XYZ',repeat=len(obs[0]))]
    chosen=[]; evaluations=0
    for _ in range(budget):
        scored=[]
        for b in bank:
            if b not in chosen:
                scored.append((readout_score(points,obs,chosen+[b]),b)); evaluations+=1
        chosen.append(max(scored,key=lambda x:x[0])[1])
    return chosen,evaluations

def basis_probabilities(psi,basis):
    U=np.ones((1,1),complex); H=np.array([[1,1],[1,-1]])/np.sqrt(2)
    for b in basis[::-1]:
        local=np.eye(2) if b=='Z' else H if b=='X' else H@np.diag([1,-1j])
        U=np.kron(U,local)
    pr=np.abs(psi@U.T)**2
    return pr/pr.sum(1,keepdims=True)

def estimates(psi, obs, bases, shots, reps, rng):
    """Joint multinomial samples; split for unbiased bilinear distances.
    shots is TOTAL per state/input across all bases, not per observable.
    Each observable is assigned to the first compatible setting.
    """
    half=shots//(2*len(bases)); used=2*half*len(bases)
    if half<1: raise ValueError('Insufficient shots')
    outs=[np.zeros((reps,len(obs),len(psi))) for _ in range(2)]
    assigned=set(); integers=np.arange(psi.shape[1])
    for basis in bases:
        ids=[i for i,o in enumerate(obs) if i not in assigned and compatible(o,basis)]
        assigned.update(ids)
        signs=[]
        for i in ids:
            parity=sum((integers>>q)&1 for q,a in enumerate(obs[i]) if a!='I')
            signs.append(1-2*(parity%2))
        if not ids: continue
        signs=np.array(signs)
        probs=basis_probabilities(psi,basis)
        for h in range(2):
            for j,pr in enumerate(probs):
                counts=rng.multinomial(half,pr,size=reps)
                outs[h][:,ids,j]=counts@signs.T/half
    if len(assigned)!=len(obs): raise ValueError('Uncovered observable')
    return outs,used

def unbiased_dist(halves,M):
    a,b=halves; P=a.shape[2]//2
    da=a[:,:,:P]-a[:,:,P:]; db=b[:,:,:P]-b[:,:,P:]
    return np.einsum('rop,oq,rqp->rp',da,M,db,optimize=True)
