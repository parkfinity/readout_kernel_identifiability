"""Reaggregate archived category means; no image inference, training or raw-score replay."""
from pathlib import Path
import json
import sys
import numpy as np

root = Path(__file__).resolve().parent.parent
saved = json.loads((root/'reference_results/ad_matched_cohort.json').read_text())
checks = {}
for name in ('full_image','ad2_image','ad2_grouped'):
    s = saved[name]
    rows = [s['category_means'][key] for key in sorted(s['category_means'])]
    for metric in ('before','after','delta'):
        assert np.isclose(np.mean([r[metric] for r in rows]),s['mean'][metric],rtol=0,atol=1e-14)
    v = np.array([r['delta'] for r in rows])
    ix = np.random.default_rng(20260927).integers(0,len(v),(10000,len(v)))
    ci = np.quantile(v[ix].mean(axis=1),[.025,.975])
    assert np.allclose(ci,s['delta_ci95'],rtol=0,atol=1e-14)
    checks[name] = dict(categories=len(rows),means=s['mean'],ci95=ci.tolist())
a,b = (saved[x]['category_means'] for x in ('ad2_image','ad2_grouped'))
v = np.array([b[k]['delta']-a[k]['delta'] for k in sorted(a)])
ix = np.random.default_rng(20260927).integers(0,len(v),(10000,len(v)))
ci = np.quantile(v[ix].mean(axis=1),[.025,.975])
assert np.allclose(ci,saved['matched_cohort_contrast']['ci95'],rtol=0,atol=1e-14)
assert np.isclose(v.mean(),saved['matched_cohort_contrast']['mean'],rtol=0,atol=1e-14)
out = dict(passed=True,checks=checks,scope=__doc__)
Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
