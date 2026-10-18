"""Bounded additional audit: twelve source settings and a closed-form witness."""
import argparse, contextlib, copy, hashlib, io, json, os, platform, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'core'))
import numpy as np
import qiskit
from qiskit.quantum_info import Operator, Pauli
import reference as ref
from pipeline import configurations, prepare, program, boundary_transport
from structural_core import Gate
from source_host import construct, verify_sources
from adapter import adapt, encode, source_features, source_states
from engines import kernel, features, objective_gradient
from normalizer import normalize, delete_common_isometries, resources
from checker import check, audit


def save(p,v): p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def err(a,b): return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


def matrix(info,E,theta,row=0):
    """Independent dense evaluator in Qiskit's little-endian basis."""
    from engines import angle_factors
    from normalizer import frame_circuit
    d=2**info['n']; U=np.eye(d,dtype=complex)
    for g in info['program']:
        if g['op']=='R':
            angle=angle_factors(g,E,theta)[0][row]
            P=Pauli(g['pauli'][::-1]).to_matrix()
            V=np.cos(angle/2)*np.eye(d)-1j*np.sin(angle/2)*P
        else: V=Operator(frame_circuit([g],info['n'])).data
        U=V@U
    return U


def phase_error(a,b):
    v=np.vdot(b,a); z=v/abs(v) if abs(v) else 1
    return err(a,z*b)


def closed_kernel(beta,X):
    phi=np.arccos(X); a=beta[0]*phi[:,0]; b=beta[1]*phi[:,1]
    c=beta[2]*phi[:,0]+beta[3]; z=np.cos(c)
    d=2*(z[:,None]**2+z[None,:]**2)-2*z[:,None]*z[None,:]*(np.cos(a[:,None]-a[None,:])+np.cos(b[:,None]-b[None,:]))
    return np.exp(-.5*d),d


def closed_gradient(beta,X,y):
    phi=np.arccos(X); a=beta[0]*phi[:,0]; b=beta[1]*phi[:,1]
    c=beta[2]*phi[:,0]+beta[3]; z=np.cos(c)
    da=phi[:,0,None]-phi[None,:,0]; db=phi[:,1,None]-phi[None,:,1]
    aa=a[:,None]-a[None,:]; bb=b[:,None]-b[None,:]
    C=np.cos(aa)+np.cos(bb); K,_=closed_kernel(beta,X)
    dz=-4*z[:,None]*np.sin(c[:,None])+2*np.sin(c[:,None])*z[None,:]*C
    J=np.stack([2*z[:,None]*z[None,:]*np.sin(aa)*da,
                2*z[:,None]*z[None,:]*np.sin(bb)*db,
                dz*phi[:,0,None]+dz.T*phi[None,:,0], dz+dz.T])
    score,S=ref.alignment(K,y)
    return score,np.einsum('ij,pij->p',S,-.5*K[None]*J)


def closed_audit(cfg,out):
    obj,q,x,t,_=construct(cfg,2,2); g,expr,meta=adapt(q,x,t)
    obs,w=ref.readout(2,'local_xyz'); info=ref.make_info(g,2,8,expr,obs,w)
    A=np.zeros((4,8)); A[0,0]=A[1,1]=A[1,5]=A[2,4]=A[3,6]=1
    rng=np.random.RandomState(90210); maxima=dict(unitary=0.,distance=0.,kernel=0.,gradient=0.)
    xs=[];thetas=[]
    for j in range(1000):
        X=rng.uniform(-.95,.95,(2,2)); theta=rng.uniform(-2,2,8); phi=np.arccos(X)
        a=theta[0]*phi[0,0]+theta[2]; b=(theta[1]+theta[5])*phi[0,1]+theta[3]+theta[7]; c=theta[4]*phi[0,0]+theta[6]
        U=np.eye(4,dtype=complex)
        for word,angle in [('XI',a),('IX',b),('XX',c)]:
            U=(np.cos(angle/2)*np.eye(4)-1j*np.sin(angle/2)*Pauli(word[::-1]).to_matrix())@U
        native=Operator(obj.get_circuit(X[0],theta)).data
        maxima['unitary']=max(maxima['unitary'],phase_error(U,native))
        f=source_features(source_states(obj,X,theta),obs)
        kc,d=closed_kernel(A@theta,X)
        maxima['distance']=max(maxima['distance'],err(d,np.sum((f[:,None]-f[None,:])**2,axis=-1)))
        maxima['kernel']=max(maxima['kernel'],err(kc,ref.gram(f,w)))
        xs.append(X);thetas.append(theta)
    X=rng.uniform(-.8,.8,(12,2)); y=rng.normal(size=12);theta=rng.uniform(.2,.8,8)
    E=encode(expr,X); matched=A@theta; naive=matched.copy(); trace=[]
    for j in range(21):
        score,grad=objective_gradient(info,E,y,theta,'batched')
        _,gb=closed_gradient(A@theta,X,y)
        maxima['gradient']=max(maxima['gradient'],err(grad,A.T@gb))
        trace.append([err(A@theta,matched),err(kernel(info,E,theta),closed_kernel(matched,X)[0]),
                      err(kernel(info,E,theta),closed_kernel(naive,X)[0])])
        if j<20:
            theta+=.1*grad
            matched+=.1*(A@A.T)@closed_gradient(matched,X,y)[1]
            naive+=.1*closed_gradient(naive,X,y)[1]
    np.savez_compressed(out/'closed_arrays.npz',X=np.array(xs),theta=np.array(thetas),train_X=X,y=y,trace=trace,A=A)
    maxima['matched_parameter_trace']=max(r[0] for r in trace)
    maxima['matched_kernel_trace']=max(r[1] for r in trace)
    return {'samples':1000,'max_errors':maxima,'naive_final_kernel_difference':trace[-1][2],
            'beta_map':A.tolist(),'optimizer':'20 alignment-ascent steps, eta=0.1; beta metric A A^T',
            'passed':max(maxima.values())<1e-10}


def run(out):
    out.mkdir(parents=True,exist_ok=False); started=time.perf_counter()
    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer): np.show_config()
    cpu='unavailable'
    try:
        cmd=['sysctl','-n','machdep.cpu.brand_string'] if sys.platform=='darwin' else ['lscpu']
        cpu=subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT).strip()
    except (OSError,subprocess.SubprocessError): pass
    save(out/'environment.json',{'python':sys.version,'numpy':np.__version__,'qiskit':qiskit.__version__,
        'platform':platform.platform(),'processor':platform.processor(),'cpu_detail':cpu,'blas':buffer.getvalue(),
        'threads':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
        'source_commit':verify_sources()['commit'],'scope':'12 existing configurations, local XYZ; no timing claim'})
    rows=[]
    for cfg in configurations():
        folder=out/cfg['id'];folder.mkdir();obj,q,x,t,_=construct(cfg,cfg['n'],cfg['layers'])
        old=prepare(cfg,'boundary');full=prepare(cfg,'transport'); compiler=prepare(cfg,'compiler_fused')
        glob,proof=normalize(compiler); pruned,deletions=delete_common_isometries(glob)
        rawg=tuple(Gate(**v) for v in old['pre_certificate_gates'])
        # Rebuild survey provenance from the exact same compilation.
        cq=ref.transpile(q,optimization_level=3,basis_gates=ref.BASIS,seed_transpiler=71,approximation_degree=1.,num_processes=1)
        cg,ce,cm=adapt(cq,x,t);tail,removed=ref.isometry_tail(cg,old['observables'],old['weights'])
        bp=[p for i,p in enumerate(cm['gate_provenance']) if i not in removed]
        boundary,bcert=boundary_transport(tail,old['observables'],old['weights'],bp)
        survey=ref.make_info(boundary,cfg['n'],len(t),ce,old['observables'],np.array(old['weights']))
        arms={'compiler':compiler,'old_boundary':old,'old_transport':full,'global_unitary':glob,'global_kernel':pruned}
        arms['global_on_boundary']=normalize(old)[0]
        arms['global_on_transport']=normalize(full)[0]
        check(arms,{'normalization':proof,'deletions':deletions})
        save(folder/'programs.json',arms); save(folder/'rewrite_record.json',{'normalization':proof,'deletions':deletions})
        maxima=dict(unitary=0.,feature=0.,kernel=0.,gradient=0.,kernel_fd=0.)
        rng=np.random.RandomState(331); allX=[];alltheta=[]
        for probe in range(20):
            X=rng.uniform(-.8,.8,(5,2));theta=rng.uniform(-1,1,len(t));y=rng.normal(size=5)
            E=encode(compiler['data_expressions'],X)
            u=Operator(obj.get_circuit(X[0],theta)).data
            maxima['unitary']=max(maxima['unitary'],phase_error(u,matrix(glob,E,theta)))
            f=source_features(source_states(obj,X,theta),old['observables'])
            maxima['feature']=max(maxima['feature'],err(f,features(glob,E,theta)[0]))
            K=ref.gram(f,old['weights']);grad=objective_gradient(compiler,E,y,theta,'batched')[1]
            for name,arm in arms.items():
                maxima['kernel']=max(maxima['kernel'],err(K,kernel(arm,E,theta)))
                maxima['gradient']=max(maxima['gradient'],err(grad,objective_gradient(arm,E,y,theta,'batched')[1]))
            if probe==0:
                af,aj,_=features(pruned,E,theta,list(range(len(t))))
                kj=ref.kernel_jacobian(af,aj,old['weights'])
                for j in range(len(t)):
                    p=theta.copy();m=theta.copy();p[j]+=1e-6;m[j]-=1e-6
                    fd=(kernel(compiler,E,p)-kernel(compiler,E,m))/2e-6
                    maxima['kernel_fd']=max(maxima['kernel_fd'],err(kj[j],fd))
            allX.append(X);alltheta.append(theta)
        np.savez_compressed(folder/'probes.npz',X=allX,theta=alltheta)
        row={'id':cfg['id'],'metrics':{k:resources(v) for k,v in arms.items()},'max_errors':maxima,
             'full_vs_survey_boundary_same_program':old['program']==survey['program'],
             'full_vs_survey_boundary_same_ids':old['ids']==survey['ids'],
             'full_vs_survey_boundary_same_expressions':old['data_expressions']==survey['data_expressions'],
             'passed':max(v for k,v in maxima.items() if k!='kernel_fd')<1e-9 and maxima['kernel_fd']<1e-6}
        save(folder/'result.json',row);rows.append(row);print(row,flush=True)
    cfg=next(c for c in configurations() if c['id']=='chebyshev_rx_n2_l2')
    closed=closed_audit(cfg,out)
    checked=audit(out)
    summary={'passed':all(r['passed'] for r in rows) and closed['passed'] and checked['passed'],
             'checker':checked,'rows':rows,'closed_form':closed,
             'elapsed_seconds':time.perf_counter()-started,'no_wall_time_or_QPU_savings_claim':True}
    save(out/'summary.json',summary)
    save(out/'manifest.json',{'files':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()],
                             'code':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in sorted(ROOT.rglob('*.py')) if 'results' not in p.parts]})
    print(json.dumps({'output':str(out),'passed':summary['passed'],'closed_form':closed},indent=2))
    return summary['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
    out=a.output or ROOT/'results'/('global_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    sys.exit(0 if run(out) else 1)
