import numpy as np, itertools
from math import comb

def ry(t): c,s=np.cos(t/2),np.sin(t/2); return np.array([[c,-s],[s,c]],dtype=complex)
def rz(t): return np.array([[np.exp(-1j*t/2),0],[0,np.exp(1j*t/2)]],dtype=complex)
def rx(t): c,s=np.cos(t/2),np.sin(t/2); return np.array([[c,-1j*s],[-1j*s,c]],dtype=complex)

def apply_1q(psi,U,q,n):
    psi=psi.reshape([2]*n); psi=np.tensordot(U,psi,axes=([1],[q])); psi=np.moveaxis(psi,0,q); return psi.reshape(-1)
def apply_cx(psi,c,t,n):
    psi=psi.reshape([2]*n); idx=[slice(None)]*n; idx[c]=1
    sub=psi[tuple(idx)]; sub=np.flip(sub,axis=(t if t<c else t-1)); psi[tuple(idx)]=sub; return psi.reshape(-1)

def enc_gates(n):
    g=[]; iq=0; ioe=0; icheb=1
    for _ in range(n):
        for _ in range(2):
            g.append((iq%n, ioe%2, 2.0*icheb)); iq+=1; ioe+=1
        icheb+=1
    return g

def build_state(n,x,theta):
    psi=np.zeros(2**n,dtype=complex); psi[0]=1
    for (q,f,c) in enc_gates(n): psi=apply_1q(psi,ry(c*np.arccos(x[f])),q,n)
    off=0
    for q in range(n):
        psi=apply_1q(psi,rz(theta[off]),q,n); psi=apply_1q(psi,rx(theta[off+1]),q,n); psi=apply_1q(psi,rz(theta[off+2]),q,n); off+=3
    for start in (0,1):
        for i in range(start,n-1,2): psi=apply_cx(psi,i,i+1,n)
    return psi

def pauli_masks(s):
    n=len(s); x=0; z=0
    for i,ch in enumerate(s):
        if ch in 'XY': x|=(1<<(n-1-i))
        if ch in 'ZY': z|=(1<<(n-1-i))
    import math
    ph=1j**bin(x&z).count('1')
    return x,z,ph

def readout(n,kind):
    if kind=='local_xyz':
        out=[]
        for q in range(n):
            for P in 'XYZ':
                s=['I']*n; s[q]=P; out.append(''.join(s))
        return out
    k=int(kind.split('_')[-1]); out=[]
    for w in range(1,k+1):
        for sites in itertools.combinations(range(n),w):
            for a in itertools.product('XYZ',repeat=w):
                s=['I']*n
                for st,P in zip(sites,a): s[st]=P
                out.append(''.join(s))
    return out

def expvals(psi, masks, idxarr, signcache):
    # returns real expectation for each observable
    out=np.empty(len(masks))
    conj=np.conjugate(psi)
    for k,(x,z,ph) in enumerate(masks):
        out[k]=np.real(ph*np.sum(signcache[k]*conj*psi[idxarr^x]))
    return out

def kernel_rank(n,kind,n_pairs=120,seed=3,tol=1e-7):
    ops=readout(n,kind); masks=[pauli_masks(s) for s in ops]
    N=2**n; idxarr=np.arange(N)
    signcache=[((-1.0)**(np.array([bin(b & z).count('1') for b in range(N)]))) for (x,z,ph) in masks]
    K=3*n; rng=np.random.default_rng(seed)
    theta=rng.uniform(-2*np.pi,2*np.pi,size=K)
    xs=rng.uniform(-0.95,0.95,size=(n_pairs,2)); xps=rng.uniform(-0.95,0.95,size=(n_pairs,2)); allX=list(xs)+list(xps)
    def feats(th):
        F=np.empty((len(allX),len(ops)))
        for p,xx in enumerate(allX): F[p]=expvals(build_state(n,xx,th),masks,idxarr,signcache)
        return F
    F0=feats(theta); d0=F0[:n_pairs]-F0[n_pairs:]
    J=np.empty((n_pairs,K)); sh=np.pi/2
    for j in range(K):
        tp=theta.copy(); tp[j]+=sh; tm=theta.copy(); tm[j]-=sh
        dF=0.5*(feats(tp)-feats(tm)); dd=dF[:n_pairs]-dF[n_pairs:]
        J[:,j]=2*np.sum(d0*dd,axis=1)
    sv=np.linalg.svd(J,compute_uv=False)
    smax=sv[0] if sv.size else 0.0
    # absolute + relative threshold (handles r=0 where smax~1e-14)
    rank=int(np.sum(sv>max(tol*smax,1e-9)))
    return rank,sv,len(ops)

if __name__=='__main__':
    import sys,time
    for (n,kind) in eval(sys.argv[1]):
        t0=time.time(); r,sv,m=kernel_rank(n,kind)
        tgt=0 if kind==f'weight_le_{n}' else 3*n-2
        print(f"n={n} {kind:12s} m={m:4d} r={r:3d} target={tgt:3d} {'OK' if r==tgt else 'DIFF'} top={sv[0]:.2f} sv[r-1]={sv[r-1]:.2e} sv[r]={sv[r] if r<len(sv) else 0:.1e} ({time.time()-t0:.1f}s)",flush=True)
