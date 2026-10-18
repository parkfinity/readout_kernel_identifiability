"""Matched generic IR interpreters. Simulator counters are not QPU resources."""
import numpy as np
from structural_core import Gate,apply_pauli,apply_clifford

def alignment(K,y):
    def center(A):return A-A.mean(0,keepdims=True)-A.mean(1,keepdims=True)+A.mean()
    C=center(K);v=y-y.mean();T=np.outer(v,v);a,b=np.linalg.norm(C),np.linalg.norm(T)
    if min(a,b)<=1e-14:raise ValueError('Degenerate centered alignment')
    score=float(np.sum(C*T)/(a*b))
    return score,center(T/(a*b)-score*C/a**2)

def angle_factors(g,X,theta):
    angle=np.zeros(len(X));factors={}
    for coefficient,ti,xi in g['terms']:
        factor=coefficient*(X[:,xi] if xi>=0 else np.ones(len(X)))
        angle+=factor*(theta[ti] if ti>=0 else 1.)
        if ti>=0 and coefficient!=0:factors[ti]=factors.get(ti,0)+factor
    return angle,factors

def cliff(state,g,n):return apply_clifford(state,Gate(g['op'],wires=tuple(g['wires'])),n)

def stats():
    return {k:0 for k in ['gradient_evaluator_calls','parameter_derivative_entries',
        'forward_state_sweeps','tangent_lane_sweeps','reverse_passes','kernel_builds',
        'state_gate_updates','tangent_gate_updates','reverse_gate_updates']}

def features(info,X,theta,ids=(),taped=False):
    n=info['n'];m=len(X);d=2**n;p=len(ids);lookup={j:i for i,j in enumerate(ids)}
    state=np.zeros((m,d),complex);state[:,0]=1
    tangent=np.zeros((p,m,d),complex);tape=[state] if taped else []
    for g in info['program']:
        if g['op'] in ('H','CX'):
            state=cliff(state,g,n)
            if p:tangent=cliff(tangent.reshape(p*m,d),g,n).reshape(p,m,d)
        else:
            angle,factors=angle_factors(g,X,theta);co=np.cos(angle[:,None]/2);si=np.sin(angle[:,None]/2)
            old=state;ps=apply_pauli(old,g['pauli']);state=co*old-1j*si*ps
            if p:
                tangent=co[None]*tangent-1j*si[None]*apply_pauli(tangent.reshape(p*m,d),g['pauli']).reshape(p,m,d)
                for ti,factor in factors.items():
                    if ti in lookup:tangent[lookup[ti]]+=factor[:,None]*(-.5*si*old-.5j*co*ps)
        if taped:tape.append(state)
    f=[];df=[]
    for o in info['observables']:
        ps=apply_pauli(state,o);f.append(np.real(np.sum(state.conj()*ps,axis=1)))
        df.append(2*np.real(np.sum(tangent.conj()*ps[None],axis=2)))
    return np.column_stack(f),np.stack(df,axis=-1),tape

def gram(f,w):return np.exp(-.5*np.sum(np.asarray(w)*(f[:,None]-f[None,:])**2,axis=-1))

def account(s,info,m,forward=0,tangent=0,reverse=0,calls=0,entries=0,kernels=0):
    for key,value in [('gradient_evaluator_calls',calls),('parameter_derivative_entries',entries),
        ('forward_state_sweeps',forward),('tangent_lane_sweeps',tangent),('reverse_passes',reverse),
        ('kernel_builds',kernels),('state_gate_updates',forward*m*len(info['program'])),
        ('tangent_gate_updates',tangent*m*len(info['program'])),
        ('reverse_gate_updates',reverse*m*len(info['program']))]:s[key]+=value

def kernel(info,X,theta,s=None):
    if s is not None:account(s,info,len(X),forward=1,kernels=1)
    return gram(features(info,X,theta)[0],info['weights'])

def objective_gradient(info,X,y,theta,engine,s=None):
    if s is None:s=stats()
    ids=info['ids'];grad=np.zeros(info['count']);w=np.asarray(info['weights'])
    if engine=='serial':
        K=kernel(info,X,theta,s);score,S=alignment(K,y)
        for j in ids:
            f,df,_=features(info,X,theta,[j]);delta=f[:,None]-f[None,:];dd=df[0,:,None]-df[0,None,:]
            grad[j]=np.sum(S*(-gram(f,w)*np.sum(w*delta*dd,axis=2)))
            account(s,info,len(X),forward=1,tangent=1,calls=1,entries=1,kernels=1)
        return score,grad
    f,df,tape=features(info,X,theta,ids if engine=='batched' else (),taped=engine=='adjoint')
    K=gram(f,w);score,S=alignment(K,y)
    sensitivity=-np.sum(((S+S.T)*K)[:,:,None]*(f[:,None]-f[None,:])*w,axis=1)
    if engine=='batched':
        grad[ids]=np.einsum('pno,no->p',df,sensitivity)
        account(s,info,len(X),forward=1,tangent=len(ids),calls=1,entries=len(ids),kernels=1)
    elif engine=='adjoint':
        lam=np.zeros_like(tape[-1])
        for j,o in enumerate(info['observables']):lam+=sensitivity[:,j,None]*apply_pauli(tape[-1],o)
        for k in range(len(info['program'])-1,-1,-1):
            g=info['program'][k]
            if g['op'] in ('H','CX'):lam=cliff(lam,g,info['n']);continue
            angle,factors=angle_factors(g,X,theta)
            if factors:
                der=np.real(np.sum(lam.conj()*(-1j*apply_pauli(tape[k+1],g['pauli'])),axis=1))
                for ti,factor in factors.items():grad[ti]+=np.sum(der*factor)
            co=np.cos(angle[:,None]/2);si=np.sin(angle[:,None]/2)
            lam=co*lam+1j*si*apply_pauli(lam,g['pauli'])
        account(s,info,len(X),forward=1,reverse=1,calls=1,entries=len(ids),kernels=1)
    else:raise ValueError(engine)
    return score,grad
