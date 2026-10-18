"""Read-only archive ingestion: validate hashes, stored pairs and timing arithmetic.
No execution of attached Python code; no new simulation/timing replication.
"""
import argparse, collections, functools, hashlib, io, json, zipfile
from pathlib import Path, PurePosixPath
import numpy as np

ROOT=Path(__file__).resolve().parent
LIMIT={'kernel':1e-10,'prediction':1e-7,'theta':1e-9,'gradient':1e-9,'objective':1e-10}

def digest(b):return hashlib.sha256(b).hexdigest()
def delta(a,b):
    x,y=np.asarray(a),np.asarray(b)
    assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
    return float(np.max(np.abs(x-y))) if x.size else 0.

def compare(a,ka,b,kb):
    assert len(a['trace'])==len(b['trace'])==10
    return {'kernel':delta(ka,kb),'prediction':delta(a['prediction'],b['prediction']),
        'theta':max(delta(a['theta_final'],b['theta_final']),max(delta(x['theta'],y['theta']) for x,y in zip(a['trace'],b['trace']))),
        'gradient':max(delta(x['gradient'],y['gradient']) for x,y in zip(a['trace'],b['trace'])),
        'objective':max(delta(x['objective'],y['objective']) for x,y in zip(a['trace'],b['trace']))}

def shift_tasks(program):
    total=0
    for g in program:
        if g['op']!='R':continue
        factors=collections.defaultdict(float)
        for coef,ti,xi in g['terms']:
            if ti>=0:factors[ti,xi]+=coef
        total+=any(v!=0 for v in factors.values())
    return total

def descriptive(rows,key='percent'):
    x=np.array([r[key] for r in rows]);b=sum(r['boundary_seconds'] for r in rows);t=sum(r['transport_seconds'] for r in rows)
    return {'pairs':len(rows),'median':float(np.median(x)),'q25':float(np.percentile(x,25)),
        'q75':float(np.percentile(x,75)),'min':float(x.min()),'max':float(x.max()),
        'positive_pairs':int(sum(x>0)),'aggregate_elapsed_reduction':100*(1-t/b)}

def block_interval(rows):
    # Keep the three datasets fixed, resample 5 seed blocks inside each one.
    # All five timing repetitions follow the selected block. Conditional descriptive
    # sensitivity only: repeated splits overlap, not a population confidence interval.
    rng=np.random.RandomState(81273);datasets=sorted({r['dataset'] for r in rows})
    by={d:[] for d in datasets}
    for d in datasets:
        for seed in sorted({r['seed'] for r in rows if r['dataset']==d}):
            by[d].append([r['percent'] for r in rows if r['dataset']==d and r['seed']==seed])
    values=[]
    for _ in range(4000):
        vals=[]
        for d in datasets:
            blocks=by[d]
            for i in rng.randint(0,len(blocks),len(blocks)):vals.extend(blocks[i])
        values.append(np.median(vals))
    return np.percentile(values,[2.5,97.5]).tolist()

def main(path):
    failures=[];maxima={k:0. for k in LIMIT};counts=collections.Counter();cold=[];reuse=[]
    with zipfile.ZipFile(path) as z:
        names=z.namelist();assert len(names)==len(set(names))
        for n in names:assert not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts
        run_names=[n for n in names if n.endswith('/run.json') and '/source_snapshot/' not in n]
        assert len(run_names)==1;prefix=run_names[0][:-len('run.json')]
        def read(name):return json.loads(z.read(prefix+name))
        run=read('run.json');cfg=run['config'];summary=read('summary.json');server=read('audit.json')
        assert cfg['mode']=='full' and cfg['design_revision']=='20261001_boundary_fused_v3'
        assert cfg['steps']==10 and cfg['train']==cfg['test']==64
        assert summary['passed'] and server['passed']
        mf=read('manifest.json')['files']
        for e in mf:assert digest(z.read(prefix+e['path']))==e['sha256'],e['path']
        counts['hash_verified_files']=len(mf)
        listed={prefix+e['path'] for e in mf}|{prefix+'manifest.json'}
        assert set(names)==listed,'Unlisted archive members'
        print('HASHES_OK',len(mf),flush=True)
        # Compare executable source and frozen input bytes with preserved local release.
        local=ROOT.parent/'interior_incremental_v1';matched=[];different=[]
        for n in names:
            marker=prefix+'source_snapshot/'
            if n.startswith(marker):
                rel=n[len(marker):];p=local/rel
                if p.is_file():
                    (matched if digest(p.read_bytes())==digest(z.read(n)) else different).append(rel)
        assert not different,different
        infos={}
        for c in cfg['configs']:
            v=read('validation/'+c['id']+'.json');assert v['passed']
            infos[c['id']]={m['method']:m['info'] for m in v['methods']}
        @functools.lru_cache(maxsize=128)
        def arr(name):
            with np.load(io.BytesIO(z.read(prefix+name)),allow_pickle=False) as f:a=f['K']
            assert np.isfinite(a).all();return a
        def array(e):
            a=arr(e['path']);assert list(a.shape)==e['shape']
            assert digest(np.ascontiguousarray(a).tobytes())==e['raw_sha256']
            counts['array_references_verified']+=1;return a
        def preserve(a,ka,b,kb):
            er=compare(a,ka,b,kb)
            for k,v in er.items():
                maxima[k]=max(maxima[k],v)
                assert v<=LIMIT[k],(k,v)
        for n in sorted(names):
            if not n.startswith(prefix+'cold/'):continue
            r=json.loads(z.read(n));assert r['passed'];rs=r['results'];aa={m:array(e) for m,e in r['arrays'].items()}
            assert set(rs)==set(cfg['methods'])
            for m in cfg['methods']:
                preserve(rs['raw'],aa['raw'],rs[m],aa[m])
                a=rs[m];info=infos[r['config_id']][m];p=len(info['ids']);g=len(info['program']);st=a['stats']
                assert a['active_parameters']==p and a['execution_operations']==g
                lane=10*p if r['engine']!='adjoint' else 0
                calls=10*p if r['engine']=='serial' else 10
                assert st['gradient_evaluator_calls']==calls and st['parameter_derivative_entries']==10*p
                assert st['tangent_gate_updates']==lane*64*g
                assert st['reverse_gate_updates']==(10*64*g if r['engine']=='adjoint' else 0)
                assert a['total_seconds']>=a['setup']['prepare_seconds']>=0
                assert a['total_seconds']>=a['train_seconds']+a['predict_seconds']>0
            b,t=rs['boundary'],rs['transport'];v=100*(1-t['total_seconds']/b['total_seconds'])
            assert abs(v-r['paired_percent_reduction'])<1e-9
            cold.append({k:r[k] for k in ('id','config_id','dataset','seed','repetition','engine')})
            cold[-1].update({'percent':v,'boundary_seconds':b['total_seconds'],'transport_seconds':t['total_seconds'],
                'boundary_first':r['execution_order'].index('boundary')<r['execution_order'].index('transport'),
                'extra_certificate_ms':1000*t['setup']['certificate_seconds'],
                'boundary_stats':b['stats'],'transport_stats':t['stats'],
                'new_inactive_symbols':b['active_parameters']-t['active_parameters']})
            counts['cold_blocks']+=1
            if counts['cold_blocks']%500==0:print('COLD',counts['cold_blocks'],flush=True)
        for n in sorted(names):
            if not n.startswith(prefix+'reuse/'):continue
            r=json.loads(z.read(n));assert r['passed'];arms=r['methods'];assert set(arms)=={'boundary','transport'}
            for i in range(20):
                preserve(arms['boundary']['fits'][i],array(r['arrays']['boundary_'+str(i)]),
                         arms['transport']['fits'][i],array(r['arrays']['transport_'+str(i)]))
            values={}
            for arm in arms:
                a=arms[arm];prev=0.
                for h in cfg['reuse_horizons']:
                    now=a['cumulative_seconds'][str(h)];assert now>prev;prev=now
                    assert now>=a['setup_seconds']+sum(f['fit_seconds'] for f in a['fits'][:h])-1e-8
            for h in cfg['reuse_horizons']:
                b=arms['boundary']['cumulative_seconds'][str(h)];t=arms['transport']['cumulative_seconds'][str(h)]
                v=100*(1-t/b);assert abs(v-r['percent_reductions'][str(h)])<1e-9
                values[str(h)]={'percent':v,'boundary_seconds':b,'transport_seconds':t}
            reuse.append({**{k:r[k] for k in ('id','config_id','dataset','seed','repetition','engine')},'horizons':values})
            counts['reuse_blocks']+=1
        assert counts['cold_blocks']==2700 and counts['reuse_blocks']==90 and counts['array_references_verified']==22500
        assert len({(r['config_id'],r['dataset'],r['seed'],r['repetition'],r['engine']) for r in cold})==2700
        all_summaries=[];primary=[];warm=[]
        for c in cfg['configs']:
            for engine in cfg['engines']:
                rows=[r for r in cold if r['config_id']==c['id'] and r['engine']==engine];assert len(rows)==75
                item={'config_id':c['id'],'engine':engine,**descriptive(rows)}
                saved=next(s for s in summary['summaries'] if s['config_id']==c['id'] and s['engine']==engine)
                assert abs(saved['median_percent']-item['median'])<1e-9
                all_summaries.append(item)
                if c['module']=='chebyshev_rx' and c['layers']==2:
                    item={**item,'conditional_block_bootstrap_interval':block_interval(rows),
                        'datasets':{d:descriptive([r for r in rows if r['dataset']==d]) for d in cfg['datasets']},
                        'order_strata':{str(o):descriptive([r for r in rows if r['boundary_first']==o]) for o in (False,True)},
                        'median_total_ms':[1000*float(np.median([r[k] for r in rows])) for k in ('boundary_seconds','transport_seconds')],
                        'median_certificate_ms':float(np.median([r['extra_certificate_ms'] for r in rows])),
                        'counters':{a:rows[0][a+'_stats'] for a in ('boundary','transport')},
                        'structure':{m:{'parameters':len(infos[c['id']][m]['ids']), 'operations':len(infos[c['id']][m]['program']),
                            'trainable_fused_angle_tasks':shift_tasks(infos[c['id']][m]['program'])} for m in ('boundary','transport')}}
                    primary.append(item)
                    wr=[r for r in reuse if r['config_id']==c['id'] and r['engine']==engine]
                    warm.append({'config_id':c['id'],'engine':engine,'horizons':{str(h):descriptive([r['horizons'][str(h)] for r in wr]) for h in cfg['reuse_horizons']},
                        'datasets_at_h20':{d:descriptive([r['horizons']['20'] for r in wr if r['dataset']==d]) for d in cfg['datasets']}})
        output={'passed':True,'archive':str(path),'archive_sha256':digest(path.read_bytes()),'counts':dict(counts),
            'max_saved_pair_errors':maxima,'server_reported_replay':server,
            'local_scope':'Hash validation, saved kernel/trace/prediction comparisons, operation-counter checks, timing arithmetic. No simulator replay or timing rerun.',
            'source_files_matching_local_release':len(matched),'source_differences':different,
            'environment':run,'all_setting_summaries':all_summaries,'primary':primary,'reuse':warm,
            'interval_scope':'Exploratory 4000 seed-block bootstrap; three datasets fixed; all five repetitions kept per block. Repeated splits overlap. Not a population-generalization CI.',
            'cold_pairs':cold,'reuse_pairs':reuse}
        (ROOT/'full_analysis.json').write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps({k:output[k] for k in ('passed','counts','max_saved_pair_errors','source_files_matching_local_release')},indent=2))
        for i in primary:print(i['config_id'],i['engine'],i['median'],i['positive_pairs'],i['conditional_block_bootstrap_interval'],i['aggregate_elapsed_reduction'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);a=p.parse_args();main(a.archive)
