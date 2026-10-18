"""Secondary closure using already validated structural baseline certificates.

This analysis is distinct from the frozen bounded Fourier pass. It never treats
budget failure or a small derivative as inactivity. An absent source symbol in
a kernel-preserving rewrite is a structural certificate. The exact coefficient
rank of its gate angles bounds the kernel differential rank from above.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from experiment import save,sha,resource_audit

def run(folder):
    rows=[]
    for slot in json.loads((folder/'run.json').read_text())['scope']:
        r=json.loads((folder/(slot['id']+'.json')).read_text())
        b=r['resources']['programs']['global_plus_commutant']
        structural=set(range(r['parameter_count']))-set(b['ids'])
        exact=set(r['exact']['inactive']) if r['exact'] else set()
        active={w['parameter'] for w in r['active_witnesses']}
        assert not (structural|exact).intersection(active)
        status=['EXACT_INACTIVE' if k in exact else 'STRUCTURAL_INACTIVE' if k in structural else
                'ACTIVE_CERTIFIED' if k in active else 'UNDECIDED' for k in range(r['parameter_count'])]
        upper=min(r['r_upper'],r['resources']['metrics']['global_plus_commutant']['angle_coefficient_rank'])
        assert r['minor']['lower']<=upper
        rows.append(dict(id=slot['id'],module=slot['module'],n=slot['n'],layers=slot['layers'],readout=slot['readout'],
                         parameters=r['parameter_count'],status=status,counts=dict(Counter(status)),
                         rank_lower=r['minor']['lower'],rank_upper=upper,
                         structural_inactive=sorted(structural),exact_inactive=sorted(exact),
                         inactive_beyond_baseline=sorted(exact-structural),
                         old_not_recovered=sorted(set(slot['old_inactive'])-(exact|structural)),
                         newly_inactive_vs_old=sorted((exact|structural)-set(slot['old_inactive'])),
                         incremental_resources=r['resources']['incremental']))
    return dict(analysis='secondary structural closure, not a replacement for the bounded-pass outcomes',
                passed=True,slots=len(rows),symbols=sum(r['parameters'] for r in rows),
                counts=dict(sum((Counter(r['status']) for r in rows),Counter())),
                rank_closed_slots=sum(r['rank_lower']==r['rank_upper'] for r in rows),
                inactive_beyond_baseline=sum(len(r['inactive_beyond_baseline']) for r in rows),
                new_inactive_vs_old=sum(len(r['newly_inactive_vs_old']) for r in rows),
                old_not_recovered=sum(len(r['old_not_recovered']) for r in rows),
                rows=rows,run_manifest_sha256=sha(folder/'manifest.json'),script_sha256=sha(Path(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.folder);save(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
