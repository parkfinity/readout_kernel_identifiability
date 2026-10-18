"""Untimed source/IR/backend checks at nonzero angles, including negative readouts."""
from dataclasses import replace
import numpy as np
from adapter import adapt,encode,source_states,source_features,gram
from source_host import construct
from pipeline import prepare,readout,METHODS,ENGINES
from certificates import projected_kernel,reduce_circuit
from engines import features,objective_gradient,alignment,kernel

def validate(cfg):
    obj,qc,x,t,_=construct(cfg,cfg['n'],cfg['layers'])
    gates,expr,_=adapt(qc,x,t);rng=np.random.RandomState(812)
    X=rng.uniform(-.8,.8,(5,2));theta=rng.uniform(.2,.8,len(t));y=rng.normal(size=5)
    E=encode(expr,X);obs,w=readout(cfg['n'])
    native=source_features(source_states(obj,X,theta),obs);K=gram(native,w)
    reference,S=alignment(K,y);derivatives=[]
    for j in range(len(t)):
        derivatives.append(np.sum(S*projected_kernel(gates,E,theta,cfg['n'],obs,w,parameter=j)[1]))
    gref=np.array(derivatives);fd=[]
    for j in range(len(t)):
        plus=theta.copy();minus=theta.copy();plus[j]+=1e-6;minus[j]-=1e-6
        def score(v):return alignment(gram(source_features(source_states(obj,X,v),obs),w),y)[0]
        fd.append((score(plus)-score(minus))/2e-6)
    fd_error=float(np.max(abs(gref-fd)));assert fd_error<=2e-7,(cfg['id'],'FD',fd_error)
    rows=[];original_signatures=[]
    for method in METHODS:
        info=prepare(cfg,method);Ei=encode(info['data_expressions'],X)
        if method!='raw':original_signatures.append(info['pre_certificate_gates'])
        ke=float(np.max(abs(kernel(info,Ei,theta)-K)));assert ke<=1e-10,(cfg['id'],method,ke)
        if method in ['raw','compiler','compiler_fused']:
            assert np.max(abs(features(info,Ei,theta)[0]-native))<=1e-10
        errors={}
        for engine in ENGINES:
            score,g=objective_gradient(info,Ei,y,theta,engine)
            err=float(np.max(abs(g-gref)));assert err<=1e-9 and abs(score-reference)<=1e-10
            errors[engine]=err
        rows.append({'method':method,'kernel_error':ke,'gradient_errors':errors,'info':info})
    assert all(s==original_signatures[0] for s in original_signatures)
    negatives=[]
    for kind in ['z','anisotropic']:
        oo,ww=readout(cfg['n'],kind);red,proof=reduce_circuit(gates,oo,ww,'transport_metric')
        a=projected_kernel(gates,E,theta,cfg['n'],oo,ww)[0]
        b=projected_kernel(red,E,theta,cfg['n'],oo,ww)[0]
        error=float(np.max(abs(a-b)));assert error<=1e-10
        negatives.append({'readout':kind,'inactive':proof['inactive_parameters'],'kernel_error':error})
    return {'config_id':cfg['id'],'passed':True,'finite_difference_error':fd_error,
            'negative_readouts':negatives,'methods':rows}
