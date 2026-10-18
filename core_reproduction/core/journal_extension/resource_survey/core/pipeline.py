"""Compile first, normalize, local-tail baseline, then extra transport."""
import time
from dataclasses import asdict
import numpy as np
from qiskit import transpile
from source_host import inventory,construct
from adapter import adapt
from structural_core import Gate,pauli,commute,conjugate_pauli
from certificates import reduce_circuit,validate_readout,invariant_generator

BASIS=['h','cx','rx','ry','rz','rxx','ryy','rzz','rzx']
METHODS=['raw','compiler_fused','compiler','local_tail','boundary','transport','commutant']
ENGINES=['serial','batched','adjoint']

def configurations():
    entries={e['module']:e for e in inventory()}
    return [dict(entries[m],n=n,layers=l,id=m+'_n'+str(n)+'_l'+str(l))
            for m in ['chebyshev_rx','chebyshev_pqc','yz_cx_encoding_circuit']
            for n in [2,4] for l in [1,2]]

def readout(n,kind='xyz'):
    obs=[pauli(n,a,q) for a in ('Z' if kind=='z' else 'XYZ') for q in range(n)]
    w=np.ones(len(obs)) if kind!='anisotropic' else np.repeat([1.,2.,3.],n)
    return obs,w

def support(g):
    return {i for i,p in enumerate(g.pauli) if p!='I'} if g.op=='R' else set(g.wires)

def local_tail(circuit,obs,weights):
    n=len(obs[0]);validate_readout(obs,weights,n)
    expected,w=readout(n)
    if obs!=expected or not np.array_equal(weights,w):
        raise ValueError('Elementary local-tail rule restricted to unit-weight local XYZ')
    work=list(enumerate(circuit));removed=[]
    for i in range(len(work)-1,-1,-1):
        original,g=work[i];s=support(g)
        if g.op!='R' or g.kind!='param' or len(s)!=1:continue
        if all(not s.intersection(support(h)) or
               (len(support(h))==1 and not h.data_dependent) for _,h in work[i+1:]):
            removed.append(original);work.pop(i)
    return tuple(g for _,g in work),{'removed_indices':sorted(removed)}

def term(g):
    if g.kind=='fixed':return [g.scale*g.value,-1,-1]
    if g.kind=='param':return [g.scale,g.index,-1]
    if g.kind=='data':return [g.scale,-1,g.feature]
    if g.kind=='product':return [g.scale,g.index,g.feature]
    raise ValueError(g.kind)

def boundary_transport(circuit,obs,weights,provenance):
    """Known tail/affine effects, but never cross a distinct overlapping data gate.

    Same-native-gate affine terms may commute; other data rotations block when
    they overlap the currently transported Pauli generator. This is a restricted
    subset of the sound transport rule, not a new completeness claim.
    """
    work=list(enumerate(circuit));certificates=[];changed=True
    while changed:
        changed=False
        for i in range(len(work)-1,-1,-1):
            original,g=work[i]
            if g.op!='R' or g.kind!='param':continue
            generator=g.pauli;safe=True;steps=[]
            for later,h in work[i+1:]:
                overlap=bool(support(Gate('R',generator)).intersection(support(h)))
                same_native=provenance[original]['native_gate']==provenance[later]['native_gate']
                if h.data_dependent and overlap and not same_native:safe=False;break
                if h.op=='R' and commute(generator,h.pauli):steps.append({'index':later,'rule':'commute'})
                elif h.op in ('H','CX'):
                    generator=conjugate_pauli(generator,h);steps.append({'index':later,'rule':'clifford'})
                else:safe=False;break
            if safe and invariant_generator(generator,obs,weights,'metric')[0]:
                certificates.append({'original_index':original,'steps':steps,'output_generator':generator})
                work.pop(i);changed=True;break
    return tuple(g for _,g in work),{'certificates':certificates,
        'contract':'No overlapping data crossing except additive terms from same native gate'}

def program(gates,provenance=None,fuse_adjacent=False):
    out=[];last=None
    for index,g in enumerate(gates):
        native=provenance[index]['native_gate'] if provenance is not None else None
        if g.op!='R':out.append({'op':g.op,'wires':list(g.wires)});last=None;continue
        if provenance is not None and native==last:
            assert out[-1]['op']=='R' and out[-1]['pauli']==g.pauli
            out[-1]['terms'].append(term(g))
        else:out.append({'op':'R','pauli':g.pauli,'terms':[term(g)]})
        last=native
    if fuse_adjacent:
        fused=[]
        for g in out:
            if g['op']=='R' and fused and fused[-1]['op']=='R' and fused[-1]['pauli']==g['pauli']:
                fused[-1]['terms'].extend(g['terms'])
            else:fused.append(g)
        out=fused
    return out

def prepare(cfg,method):
    if method not in METHODS:raise ValueError(method)
    start=time.perf_counter();obj,qc,x,t,_=construct(cfg,cfg['n'],cfg['layers'])
    source_time=time.perf_counter()-start
    tick=time.perf_counter()
    if method!='raw':
        qc=transpile(qc,optimization_level=3,basis_gates=BASIS,seed_transpiler=71,
                     approximation_degree=1.,num_processes=1)
    compile_time=time.perf_counter()-tick
    tick=time.perf_counter();c,expr,meta=adapt(qc,x,t);adapt_time=time.perf_counter()-tick
    original=c;obs,w=readout(cfg['n']);local_proof=None;boundary_proof=None;proof=None;lt=ct=bt=0.
    if method in ['local_tail','boundary','transport']:
        tick=time.perf_counter();c,local_proof=local_tail(c,obs,w);lt=time.perf_counter()-tick
    if method in ['boundary','transport']:
        provenance=[v for i,v in enumerate(meta['gate_provenance']) if i not in local_proof['removed_indices']]
        tick=time.perf_counter();c,boundary_proof=boundary_transport(c,obs,w,provenance);bt=time.perf_counter()-tick
    if method in ['transport','commutant']:
        tick=time.perf_counter()
        c,proof=reduce_circuit(c,obs,w,'transport_'+('metric' if method=='transport' else 'commutant'))
        ct=time.perf_counter()-tick
    # Common cheap re-fusion prevents a benefit caused solely by artificial IR
    # splitting of adjacent same-axis rotations. raw/compiler remain split ablations.
    p=program(c,meta['gate_provenance'] if method=='compiler_fused' else None,
              fuse_adjacent=method not in ['raw','compiler'])
    ids=sorted({int(i) for g in p if g['op']=='R' for coef,i,j in g['terms'] if i>=0 and coef!=0})
    return {'config_id':cfg['id'],'n':cfg['n'],'count':len(t),'method':method,
            'program':p,'ids':ids,'data_expressions':expr,'observables':obs,'weights':w.tolist(),
            'source_seconds':source_time,'compile_seconds':compile_time,'adapt_seconds':adapt_time,
            'local_tail_seconds':lt,'boundary_seconds':bt,'certificate_seconds':ct,'prepare_seconds':time.perf_counter()-start,
            'pre_certificate_gates':[asdict(g) for g in original],
            'remaining_gates':[asdict(g) for g in c],'local_proof':local_proof,
            'boundary_proof':boundary_proof,'extra_proof':proof,
            'unknown_parameters':ids,'inactive_parameters':sorted(set(range(len(t)))-set(ids)),
            'native_operations':meta['native_operations'],'ir_gates_before':len(original),
            'ir_gates_after':len(c),'execution_operations':len(p)}
