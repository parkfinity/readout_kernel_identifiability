"""Full artifact/paired arithmetic audit plus specified independent-path replay."""
import argparse,hashlib,json,traceback,zipfile
from pathlib import Path
import numpy as np
from structural_core import Gate
from certificates import projected_kernel
from adapter import encode
from engines import alignment
from experiment import digest,write,manifest,differences,check,load_data

def load_arrays(folder,entries):
    out={}
    for key,e in entries.items():
        with np.load(folder/e['path'],allow_pickle=False) as a:K=a['K']
        assert list(K.shape)==e['shape']
        assert hashlib.sha256(np.ascontiguousarray(K).tobytes()).hexdigest()==e['raw_sha256']
        out[key]=K
    return out

def replay(info,X,y,result,K_saved,size):
    c=tuple(Gate(**g) for g in info['remaining_gates']);E=encode(info['data_expressions'],X)
    theta=np.array(result['trace'][0]['theta']);max_g=0.;max_k=0.;max_s=0.
    for row in result['trace']:
        np.testing.assert_allclose(theta,row['theta'],atol=1e-9,rtol=0)
        K=projected_kernel(c,E[:size],theta,info['n'],info['observables'],info['weights'])[0]
        score,S=alignment(K,y[:size]);grad=np.zeros(info['count'])
        for j in range(info['count']):
            dK=projected_kernel(c,E[:size],theta,info['n'],info['observables'],info['weights'],parameter=j)[1]
            grad[j]=np.sum(S*dK)
        ge=float(np.max(abs(grad-row['gradient'])));se=abs(score-row['objective'])
        max_g=max(max_g,ge);max_s=max(max_s,se);assert ge<=1e-9 and se<=1e-10
        # Saved increments avoid confusing trajectory drift with local derivative validation.
        theta+=.1*np.array(row['gradient'])
    np.testing.assert_allclose(theta,result['theta_final'],atol=1e-9,rtol=0)
    K=projected_kernel(c,E,theta,info['n'],info['observables'],info['weights'])[0]
    max_k=float(np.max(abs(K-K_saved)));assert max_k<=1e-10
    mean=float(y[:size].mean());pred=K[size:,:size]@np.linalg.solve(K[:size,:size]+.1*np.eye(size),y[:size]-mean)+mean
    pe=float(np.max(abs(pred-result['prediction'])));assert pe<=1e-7
    return {'gradient':max_g,'objective':max_s,'kernel':max_k,'prediction':pe}

def audit_run(folder):
    folder=Path(folder).resolve();error=None;report={}
    try:
        mf=json.loads((folder/'manifest.json').read_text())
        for e in mf['files']:assert digest(folder/e['path'])==e['sha256'],e['path']
        cfg=json.loads((folder/'run.json').read_text())['config'];summary=json.loads((folder/'summary.json').read_text())
        assert summary['passed']
        validations={}
        for c in cfg['configs']:
            r=json.loads((folder/'validation'/(c['id']+'.json')).read_text());assert r['passed']
            validations[c['id']]={r['method']:r['info'] for r in r['methods']}
        counts={'cold':0,'reuse':0,'arrays_checked':0,'replayed_cold':0,'replayed_reuse_fits':0}
        cold_values=[];replay_errors=[];primary=[]
        for p in sorted((folder/'cold').glob('*.json')):
            r=json.loads(p.read_text());arrays=load_arrays(folder,r['arrays']);counts['arrays_checked']+=len(arrays)
            for m in cfg['methods']:
                assert check(differences(r['results']['raw'],arrays['raw'],r['results'][m],arrays[m]))
                a=r['results'][m];assert a['total_seconds']>=a['setup']['prepare_seconds']>=0
                assert a['total_seconds']>=a['train_seconds']+a['predict_seconds']
            reduction=100*(1-r['results']['transport']['total_seconds']/r['results']['boundary']['total_seconds'])
            assert abs(reduction-r['paired_percent_reduction'])<1e-10
            cold_values.append((r['config_id'],r['engine'],reduction));counts['cold']+=1
            # One repetition, serial reference, every configuration and dataset/seed.
            if r['repetition']==0 and r['engine']=='serial':
                X,y,_=load_data(r['dataset'],r['seed'],cfg['train'])
                replay_errors.append(replay(validations[r['config_id']]['raw'],X,y,r['results']['raw'],arrays['raw'],cfg['train']))
                counts['replayed_cold']+=1
        for p in sorted((folder/'reuse').glob('*.json')):
            r=json.loads(p.read_text());arrays=load_arrays(folder,r['arrays']);counts['arrays_checked']+=len(arrays)
            for i in range(max(cfg['reuse_horizons'])):
                assert check(differences(r['methods']['boundary']['fits'][i],arrays['boundary_'+str(i)],
                                         r['methods']['transport']['fits'][i],arrays['transport_'+str(i)]))
                if r['repetition']==0 and r['engine']=='serial':
                    X,y,_=load_data(r['dataset'],r['seed'],cfg['train'])
                    replay_errors.append(replay(validations[r['config_id']]['raw'],X,y,
                        r['methods']['transport']['fits'][i],arrays['transport_'+str(i)],cfg['train']))
                    counts['replayed_reuse_fits']+=1
            for method in ['boundary','transport']:
                ts=[r['methods'][method]['cumulative_seconds'][str(h)] for h in cfg['reuse_horizons']]
                assert all(b>a for a,b in zip(ts,ts[1:]))
            for h in cfg['reuse_horizons']:
                value=100*(1-r['methods']['transport']['cumulative_seconds'][str(h)]/
                            r['methods']['boundary']['cumulative_seconds'][str(h)])
                assert abs(value-r['percent_reductions'][str(h)])<1e-10
            expected=next((h for h in cfg['reuse_horizons'] if r['percent_reductions'][str(h)]>0),None)
            assert expected==r['first_observed_break_even_h'];counts['reuse']+=1
        expected_cold=len(cfg['configs'])*len(cfg['datasets'])*len(cfg['seeds'])*cfg['repetitions']*len(cfg['engines'])
        expected_reuse=sum(c['module']=='chebyshev_rx' and c['layers']==2 for c in cfg['configs'])*len(cfg['datasets'])*cfg['repetitions']*len(cfg['engines'])
        assert counts['cold']==expected_cold==summary['cold_blocks']
        assert counts['reuse']==expected_reuse==summary['reuse_blocks']
        for row in summary['summaries']:
            v=[x[2] for x in cold_values if x[:2]==(row['config_id'],row['engine'])]
            assert len(v)==row['pairs'] and abs(float(np.median(v))-row['median_percent'])<1e-10
        report={'passed':True,'manifest_files':len(mf['files']),**counts,
            'max_replay_errors':{k:max((e[k] for e in replay_errors),default=0.) for k in ['kernel','gradient','objective','prediction']},
            'replay_scope':'All saved paired kernels/traces checked; independent scalar-source replay of rep0 serial cold reference for every dataset/seed/config, and rep0 serial reuse fits for both primary configs. Replay checks saved increments, not autonomous retraining. Timings arithmetic only.'}
    except BaseException:
        error=traceback.format_exc();report={'passed':False,'error':error};print(error,flush=True)
    write(folder/'audit.json',report);manifest(folder)
    with zipfile.ZipFile(folder.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,'results/'+folder.name+'/'+str(p.relative_to(folder)))
    print(json.dumps(report,indent=2),flush=True)
    if error:raise RuntimeError('Audit failed; evidence retained')
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');audit_run(p.parse_args().directory)
