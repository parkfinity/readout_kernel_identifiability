"""Validate saved samples and summarize paired outcomes without selecting test winners."""
import argparse
from core import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',nargs='?',type=Path);a=ap.parse_args()
    root=a.directory or max((HERE/'results').glob('*/summary.json'),key=lambda p:p.parent.name.split('_',1)[1]).parent
    summary=json.loads((root/'summary.json').read_text());plan=load_plan()
    if summary['plan_sha256']!=sha(HERE/'plan.json'):raise ValueError('Plan mismatch')
    expected=4 if summary['mode']=='pilot' else 60
    if len(summary['records'])!=expected:raise ValueError('Incomplete run')
    checks=[];groups={};vr=[];zs=[];empirical_pairs=[]
    for info in summary['records']:
        rec=json.loads((root/info['file']).read_text())
        spec=next(s for s in plan['panels'][rec['panel']]['cases'] if s['case']==rec['case'])
        checks.append(rec['arms']==spec['arms'])
        samples=np.load(root/(Path(info['file']).stem+'_samples.npz'),allow_pickle=False)
        for mc in rec['monte_carlo']:
            s=samples['d%d_%s_s%d'%(mc['direction'],mc['arm'],mc['requested_shots'])]
            checks += [np.isfinite(s).all(),len(s)==summary['replicates'],
                       np.isclose(s.mean(),mc['empirical_mean'],rtol=1e-10,atol=1e-12),
                       np.isclose(s.std(ddof=1),mc['empirical_sd'],rtol=1e-10,atol=1e-12)]
            if mc['variance_ratio'] is not None:vr.append(mc['variance_ratio'])
            if mc['bias_mc_se'] is not None:zs.append(abs(mc['bias_mc_se']))
            if mc['arm']=='shot_selected':
                for arm in ['fixed_diagonal','rank_selected']:
                    b=next(x for x in rec['monte_carlo'] if x['arm']==arm and x['direction']==mc['direction'] and x['requested_shots']==mc['requested_shots'])
                    empirical_pairs.append(dict(panel=rec['panel'],case=rec['case'],point=rec['test_point'],
                        direction=mc['direction'],shots=mc['requested_shots'],baseline=arm,
                        sd_ratio=mc['empirical_sd']/b['empirical_sd'] if b['empirical_sd'] else None,
                        snr_ratio=mc['empirical_snr']/b['empirical_snr'] if b['empirical_snr'] and b['empirical_snr']>1e-10 else None,
                        note='Each method has its own ideal matched contrast; raw SD alone is not performance.'))
        for c in info['comparisons']:
            key=(rec['panel'],rec['case'],c['shots'],c['baseline'])
            groups.setdefault(key,[]).append(c)
        samples.close()
    paired=[]
    for key,rows in groups.items():
        ratios=[r['retained_sensitivity_ratio'] for r in rows if r['retained_sensitivity_ratio'] is not None]
        paired.append(dict(panel=key[0],case=key[1],shots=key[2],baseline=key[3],points=len(rows),
            target_dimension=rows[0]['target_dimension'],
            rank_better=sum(r['new_rank']>r['baseline_rank'] for r in rows),
            rank_worse=sum(r['new_rank']<r['baseline_rank'] for r in rows),
            selected_full_rank=sum(r['new_rank']==r['target_dimension'] for r in rows),
            median_retained_sensitivity_ratio=float(np.median(ratios)) if ratios else None,
            min_retained_sensitivity_ratio=float(min(ratios)) if ratios else None,
            max_retained_sensitivity_ratio=float(max(ratios)) if ratios else None,
            median_selected_weakest=float(np.median([r['new_weakest'] for r in rows])),
            median_baseline_weakest=float(np.median([r['baseline_weakest'] for r in rows]))))
    audit=dict(passed=bool(all(checks)),mode=summary['mode'],verified_records=len(summary['records']),
        median_empirical_to_predicted_variance=float(np.median(vr)),
        variance_ratio_5_95_percentiles=np.percentile(vr,[5,95]),
        max_abs_bias_mc_se=max(zs),paired_analytic=paired,paired_empirical=empirical_pairs,
        note='Variance statistics are diagnostics, not independent hypothesis tests. Rank-deficient targets have zero weakest sensitivity. Do not average seed/theta as independent discoveries.')
    save(root/'analysis.json',audit)
    if not audit['passed']:raise ValueError('Saved statistics mismatch')
    print(json.dumps({k:audit[k] for k in ['passed','mode','verified_records','median_empirical_to_predicted_variance','max_abs_bias_mc_se']},indent=2))

if __name__=='__main__':main()
