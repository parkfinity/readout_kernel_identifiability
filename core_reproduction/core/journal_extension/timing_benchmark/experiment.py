"""Fixed-architecture cold/warm benchmark; no selection and no numerical pruning."""
import argparse,contextlib,datetime,hashlib,io,json,os,platform,shutil,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np
import qiskit,sympy
from adapter import encode
from pipeline import configurations,prepare,METHODS,ENGINES
from engines import kernel,objective_gradient,stats
from validation import validate
ROOT=Path(__file__).resolve().parent

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def load_data(name,seed,size):
    path=ROOT/'frozen_inputs'/(name+'_'+str(seed)+'.json')
    mf=json.loads((ROOT/'frozen_manifest.json').read_text())
    entry=next(e for e in mf['files'] if e['path']==str(path.relative_to(ROOT)))
    assert digest(path)==entry['sha256']
    r=json.loads(path.read_text());X=np.array(r['X']);y=np.array(r['y'])
    assert X.shape==(128,2) and y.shape==(128,) and np.isfinite(X).all() and np.max(abs(X))<1
    assert not set(r['metadata']['train_indices'])&set(r['metadata']['test_indices'])
    indices=np.r_[np.arange(size),np.arange(64,64+size)]
    return X[indices],y[indices],entry

def initial(seed,index,count,reuse=0):
    return np.random.RandomState(seed+1009*index+100000*reuse).uniform(-.3,.3,count)

def fit(info,E,y,theta0,steps,size):
    all_start=time.perf_counter();s=stats();theta=theta0.copy();trace=[]
    tick=time.perf_counter()
    for step in range(steps):
        score,g=objective_gradient(info,E[:size],y[:size],theta,info['engine'],s)
        trace.append({'theta':theta.tolist(),'gradient':g.tolist(),'objective':score})
        theta+=.1*g
    train_time=time.perf_counter()-tick;tick=time.perf_counter()
    K=kernel(info,E,theta,s);mean=float(y[:size].mean())
    alpha=np.linalg.solve(K[:size,:size]+.1*np.eye(size),y[:size]-mean)
    pred=K[size:,:size]@alpha+mean;prediction_time=time.perf_counter()-tick
    return {'theta_final':theta.tolist(),'trace':trace,'prediction':pred.tolist(),
            'train_seconds':train_time,'predict_seconds':prediction_time,
            'fit_seconds':time.perf_counter()-all_start,'stats':s},K

def cold(cfg,method,engine,X,y,seed,index,steps,size):
    start=time.perf_counter();info=prepare(cfg,method);info['engine']=engine
    tick=time.perf_counter();E=encode(info['data_expressions'],X);encoding_time=time.perf_counter()-tick
    record,K=fit(info,E,y,initial(seed,index,info['count']),steps,size)
    total=time.perf_counter()-start
    record.update({'method':method,'engine':engine,'total_seconds':total,'encoding_seconds':encoding_time,
                   'setup':{k:v for k,v in info.items() if k.endswith('_seconds')},
                   'active_parameters':len(info['ids']),'execution_operations':len(info['program']),
                   'inactive_parameters':info['inactive_parameters']})
    return record,K

def differences(a,Ka,b,Kb):
    err={'kernel':float(np.max(abs(Ka-Kb))),
         'prediction':float(np.max(abs(np.array(a['prediction'])-b['prediction']))),
         'theta':float(np.max(abs(np.array(a['theta_final'])-b['theta_final']))),
         'gradient':0.,'objective':0.}
    for x,y in zip(a['trace'],b['trace']):
        err['gradient']=max(err['gradient'],float(np.max(abs(np.array(x['gradient'])-y['gradient']))))
        err['objective']=max(err['objective'],abs(x['objective']-y['objective']))
    return err

def check(err):
    limits={'kernel':1e-10,'prediction':1e-7,'theta':1e-9,'gradient':1e-9,'objective':1e-10}
    return all(err[k]<=limits[k] for k in limits)

def save_arrays(folder,arrays):
    # Lossless content-addressed storage: exact duplicate matrices share an artifact.
    result={}
    for key,K in arrays.items():
        sha=hashlib.sha256(np.ascontiguousarray(K).tobytes()).hexdigest();name='arrays/'+sha+'.npz'
        path=folder/name
        if not path.exists():
            path.parent.mkdir(exist_ok=True);np.savez_compressed(path,K=K)
        result[key]={'path':name,'raw_sha256':sha,'shape':list(K.shape)}
    return result

def reuse_block(cfg,engine,X,y,seed,index,steps,size,horizons,order):
    rows={};arrays={}
    for method in order:
        start=time.perf_counter();info=prepare(cfg,method);info['engine']=engine
        E=encode(info['data_expressions'],X);setup=time.perf_counter()-start
        loop_start=time.perf_counter();fits=[];checkpoints={}
        for r in range(max(horizons)):
            record,K=fit(info,E,y,initial(seed,index,info['count'],r),steps,size)
            fits.append(record);arrays[method+'_'+str(r)]=K
            if r+1 in horizons:checkpoints[str(r+1)]=setup+time.perf_counter()-loop_start
        rows[method]={'setup_seconds':setup,'cumulative_seconds':checkpoints,'fits':fits}
    errors=[]
    for i in range(max(horizons)):
        errors.append(differences(rows['boundary']['fits'][i],arrays['boundary_'+str(i)],
                                  rows['transport']['fits'][i],arrays['transport_'+str(i)]))
    passed=all(check(e) for e in errors)
    reductions={str(h):100*(1-rows['transport']['cumulative_seconds'][str(h)]/
                            rows['boundary']['cumulative_seconds'][str(h)]) for h in horizons}
    return {'methods':rows,'errors':errors,'passed':passed,'percent_reductions':reductions,
            'first_observed_break_even_h':next((h for h in horizons if reductions[str(h)]>0),None)},arrays

def manifest(folder):
    entries=[{'path':str(p.relative_to(folder)),'sha256':digest(p)} for p in sorted(folder.rglob('*'))
             if p.is_file() and p.name!='manifest.json']
    write(folder/'manifest.json',{'files':entries})

def run(mode):
    now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder=ROOT/'results'/(mode+'_'+now);folder.mkdir(parents=True)
    for p in ROOT.iterdir():
        if p.is_file() and (p.suffix in ['.py','.sh','.txt','.md','.json']):
            (folder/'source_snapshot').mkdir(exist_ok=True);shutil.copy2(p,folder/'source_snapshot'/p.name)
    for name in ['vendor','frozen_inputs']:
        shutil.copytree(ROOT/name,folder/'source_snapshot'/name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    cfgs=configurations()
    if mode=='smoke':cfgs=[c for c in cfgs if c['module']=='chebyshev_rx' and c['layers']==2]
    seeds=[42,123,777,999,2024] if mode=='full' else [42]
    names=['wdbc','ionosphere','auto_mpg'] if mode=='full' else ['ionosphere']
    repetitions=5 if mode=='full' else 1;steps=10 if mode=='full' else 4 if mode=='pilot' else 2
    size=8 if mode=='smoke' else 64;horizons=[1,2,5,10,20] if mode=='full' else [1,2,5] if mode=='pilot' else [1,2]
    cfg={'mode':mode,'configs':cfgs,'datasets':names,'seeds':seeds,'repetitions':repetitions,
         'steps':steps,'train':size,'test':size,'methods':METHODS,'engines':ENGINES,
         'learning_rate':.1,'ridge_unnormalized':.1,'gamma':.5,'reuse_horizons':horizons,
         'reuse_seeds':[42],'no_model_selection':True,
         'design_revision':'20261001_boundary_fused_v3',
         'primary':'paired boundary versus transport cold total; report n2/n4 L2 ChebyshevRx separately',
         'scope':'Pinned source host and custom NumPy backends; no full sQUlearn runtime, QCAS or QPU'}
    if mode=='validation':cfg['datasets']=[]
    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):np.show_config()
    write(folder/'run.json',{'config':cfg,'python':sys.version,'platform':platform.platform(),
        'machine':platform.machine(),'processor':platform.processor(),'logical_cpu_count':os.cpu_count(),
        'numpy':np.__version__,'qiskit':qiskit.__version__,'sympy':sympy.__version__,
        'blas':buffer.getvalue(),'thread_env':{k:os.environ.get(k) for k in
            ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}})
    passed=False;rows=[];warm_rows=[];failure=None
    try:
        if qiskit.__version__!='1.2.4':raise RuntimeError('Qiskit 1.2.4 required; use requirements_py38.txt')
        for c in cfgs:
            write(folder/'validation'/(c['id']+'.json'),validate(c));print('VALIDATED',c['id'],flush=True)
        # Warm every engine/method/config using inputs independent of the timed tasks.
        rng=np.random.RandomState(71);warmX=rng.uniform(-.5,.5,(16,2));warmy=rng.normal(size=16)
        if mode!='validation':
            for c in cfgs:
                for method in METHODS:
                    info=prepare(c,method);E=encode(info['data_expressions'],warmX)
                    for engine in ENGINES:
                        info['engine']=engine;fit(info,E,warmy,initial(71,0,info['count']),1,8)
        for index,c in enumerate(cfgs):
            for name in cfg['datasets']:
                for seed in seeds:
                    X,y,entry=load_data(name,seed,size)
                    for rep in range(repetitions):
                        reference=None
                        for ei,engine in enumerate(ENGINES):
                            order=list(np.random.RandomState(917+index*101+seed+rep*7+ei).permutation(METHODS))
                            results={};arrays={}
                            for method in order:
                                results[method],arrays[method]=cold(c,method,engine,X,y,seed,index,steps,size)
                            errors={m:differences(results['raw'],arrays['raw'],results[m],arrays[m]) for m in METHODS}
                            if reference is None:reference=(results['raw'],arrays['raw'])
                            errors['cross_backend']=differences(*reference,results['raw'],arrays['raw'])
                            ok=all(check(e) for e in errors.values())
                            key=c['id']+'_'+name+'_'+str(seed)+'_r'+str(rep)+'_'+engine
                            output={'id':key,'config_id':c['id'],'dataset':name,'seed':seed,'repetition':rep,
                                'engine':engine,'execution_order':order,'input':entry,'results':results,
                                'errors':errors,'passed':ok,'arrays':save_arrays(folder,arrays),
                                'condition_number':float(np.linalg.cond(arrays['raw'][:size,:size]+.1*np.eye(size))),
                                'paired_percent_reduction':100*(1-results['transport']['total_seconds']/results['boundary']['total_seconds']),
                                'secondary_vs_local_tail_percent':100*(1-results['transport']['total_seconds']/results['local_tail']['total_seconds'])}
                            write(folder/'cold'/(key+'.json'),output)
                            rows.append({k:output[k] for k in ['id','config_id','dataset','seed','repetition','engine','passed','paired_percent_reduction']})
                            if not ok:raise AssertionError('Cold preservation failure: '+key)
                    print('COLD_DONE',c['id'],name,seed,flush=True)
            if c['module']=='chebyshev_rx' and c['layers']==2:
                for name in cfg['datasets']:
                    X,y,_=load_data(name,42,size)
                    for rep in range(repetitions):
                        for ei,engine in enumerate(ENGINES):
                            order=['boundary','transport'] if (index+rep+ei)%2==0 else ['transport','boundary']
                            result,arrays=reuse_block(c,engine,X,y,42,index,steps,size,horizons,order)
                            key=c['id']+'_'+name+'_42_r'+str(rep)+'_'+engine
                            result.update({'id':key,'config_id':c['id'],'dataset':name,'seed':42,
                                'repetition':rep,'engine':engine,'execution_order':order,'arrays':save_arrays(folder,arrays)})
                            write(folder/'reuse'/(key+'.json'),result)
                            warm_rows.append({k:result[k] for k in ['id','config_id','dataset','seed','repetition','engine','passed','percent_reductions','first_observed_break_even_h']})
                            if not result['passed']:raise AssertionError('Reuse preservation failure: '+key)
                    print('REUSE_DONE',c['id'],name,flush=True)
        expected=len(cfgs)*len(cfg['datasets'])*len(seeds)*repetitions*len(ENGINES)
        expected_reuse=sum(c['module']=='chebyshev_rx' and c['layers']==2 for c in cfgs)*len(cfg['datasets'])*repetitions*len(ENGINES)
        assert len(rows)==expected and len(warm_rows)==expected_reuse
        passed=True
    except BaseException:
        failure=traceback.format_exc();print(failure,flush=True)
    summaries=[]
    for c in cfgs:
        for engine in ENGINES:
            values=[r['paired_percent_reduction'] for r in rows if r['config_id']==c['id'] and r['engine']==engine]
            if values:summaries.append({'config_id':c['id'],'engine':engine,'pairs':len(values),
                'median_percent':float(np.median(values)),'min_percent':min(values),'max_percent':max(values)})
    write(folder/'summary.json',{'passed':passed,'failure':failure,'cold_blocks':len(rows),
        'cold_fits':len(rows)*len(METHODS),'reuse_blocks':len(warm_rows),
        'reuse_fits':len(warm_rows)*2*max(horizons),'summaries':summaries,'cold_pairs':rows,'reuse_pairs':warm_rows,
        'scope':'Technical verification for smoke/pilot; server full needed for primary performance interpretation. No independent datasets claim.'})
    manifest(folder)
    target=folder.with_suffix('.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(ROOT)))
    print(json.dumps({'passed':passed,'output':str(folder),'zip':str(target)},indent=2),flush=True)
    if not passed:raise SystemExit(1)
    return folder

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','pilot','full','validation']);run(p.parse_args().mode)
