"""v18 re-run of the block-cover direct-summand test with the constant mode included.

(DS)  rank_D[1, H_cov, H_oth] == N_cov + rank_D[1, H_oth]
i.e. the cover-mode functions are linearly independent and their span meets
span{1, H_oth} only at zero.  H_cov = data modes on the cover supports,
H_oth = all other data modes on weight-one pullback supports.

Mode construction is imported unchanged from ../kyriienko_proof_v1/analysis
(allmodes_indep.py omits the constant column; this script adds it).
Floating-point sampled ranks (absolute tolerance 1e-9, as in the proof notes):
an implementation check, not a certificate.  The proof folder is read only;
bytecode writing is disabled so no __pycache__ is created there.

Run: python3 check_ds_constant.py   -> analysis/ds_constant_check.json
"""
import contextlib, io, itertools, json, sys
from pathlib import Path
import numpy as np

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
proof = root.parent / 'kyriienko_proof_v1' / 'analysis'
sys.path.insert(0, str(proof))
with contextlib.redirect_stdout(io.StringIO()):          # clifford_structure prints on import
    from fast_sim import enc_gates
    from clifford_structure import cnot_pairs, adEdag, single

SIZES = [2, 4, 5, 6, 7, 8, 10, 12]                        # sizes reported in Table tab:kyriscan
SEEDS = [1, 7]
TOL = 1e-9

def to_str(xz):
    x, z = xz
    return ''.join({(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}[(x[i], z[i])] for i in range(len(x)))

def pullback_supports(n):
    pairs = cnot_pairs(n); S = set()
    for q in range(n):
        for P in 'XYZ':
            s = to_str(adEdag(single(n, q, P), pairs))
            S.add(tuple(i for i, ch in enumerate(s) if ch != 'I'))
    return sorted(S)

def cover_supports(n):
    s = [(2*j, 2*j+1) for j in range(n//2)]
    if n % 2 == 1:
        s.append((n-1,))
    return s

def phases(n, X):
    P = np.zeros((len(X), n))
    for (q, f, c) in enc_gates(n):
        P[:, q] += c*np.arccos(X[:, f])
    return P

def rank(M):
    s = np.linalg.svd(M, compute_uv=False)
    r = int(np.sum(s > TOL))
    return r, float(s[r-1]) if r else None, float(s[r]) if r < len(s) else 0.0

out = {'definition': 'rank_D[1,H_cov,H_oth] == N_cov + rank_D[1,H_oth]', 'tolerance_abs': TOL,
       'sampling': 'x uniform in (-0.95,0.95)^2, 3*#modes+30 points', 'seeds': SEEDS,
       'status': 'NUMERICAL (floating sampled rank), not a certificate', 'results': {}}
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    res = {}
    for n in SIZES:
        sups = pullback_supports(n)
        allm = [(S, a) for S in sups if len(S) > 0 for a in itertools.product('XZ', repeat=len(S))]
        cov = {(tuple(S), a) for S in cover_supports(n) for a in itertools.product('XZ', repeat=len(S))}
        covm = [m for m in allm if m in cov]; oth = [m for m in allm if m not in cov]
        X = rng.uniform(-0.95, 0.95, size=(3*len(allm)+30, 2)); P = phases(n, X)
        def col(m):
            S, a = m; v = np.ones(len(X))
            for q, t in zip(S, a):
                v = v*(np.sin(P[:, q]) if t == 'X' else np.cos(P[:, q]))
            return v
        one = np.ones((len(X), 1))
        Hc = np.column_stack([col(m) for m in covm]); Ho = np.column_stack([col(m) for m in oth])
        rc, kc, dc = rank(Hc)
        ro1, ko, do = rank(np.hstack([one, Ho]))
        ra1, ka, da = rank(np.hstack([one, Hc, Ho]))
        ro, _, _ = rank(Ho); ra, _, _ = rank(np.hstack([Hc, Ho]))
        res[str(n)] = dict(n_modes=len(allm), N_cov=len(covm), rank_cov=rc, rank_1_oth=ro1, rank_1_cov_oth=ra1,
                           DS=(ra1 == len(covm)+ro1), cover_deficit=len(covm)-rc, intersection_dim=rc+ro1-ra1,
                           no_constant=dict(rank_all=ra, rank_cov_plus_rank_oth=rc+ro),
                           min_kept_sv=min(kc, ko, ka), max_dropped_sv=max(dc, do, da))
    out['results'][str(seed)] = res
verdicts = {s: {n: r['DS'] for n, r in res.items()} for s, res in out['results'].items()}
assert len({json.dumps(v, sort_keys=True) for v in verdicts.values()}) == 1, 'seed-dependent verdict'
(root/'analysis/ds_constant_check.json').write_text(json.dumps(out, indent=2)+'\n')
for n, r in out['results'][str(SEEDS[0])].items():
    print(n, 'DS' if r['DS'] else 'fails', 'cover %d/%d' % (r['rank_cov'], r['N_cov']),
          'intersection', r['intersection_dim'], 'gap %.2e / %.2e' % (r['min_kept_sv'], r['max_dropped_sv']))
