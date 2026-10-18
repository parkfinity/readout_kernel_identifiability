"""Run the Fourier-Gram certification on the 30 frozen settings.

Reads (never writes) certified_dimension_extension_v1/.  Writes
results/<UTC timestamp>/ with one JSON per setting, summary.json,
environment.json, provenance.json and manifest.json.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
import traceback

import numpy as np
import sympy as sp
import mpmath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'copied_from_extension_v1'))
import fourier_gram as fg  # noqa: E402
import interval_engine as ie  # noqa: E402  (copied, unmodified)
from mpmath import iv  # noqa: E402

JE = HERE.parents[1]
ORIG = JE / 'certified_dimension_extension_v1'
FROZEN = ORIG / 'frozen_inputs'
ORIG_RUN = ORIG / 'results' / 'extension_20261001T232302_991451Z'
TEX = JE / 'manuscript_census_reference' / 'extension_results.tex'
SEED = 20261005
EXACT_RATIONAL = {  # settings where the full exact-rational Jacobian is also computed
    'kyriienko_nonlinear_encoding_circuit_n2_l1_local_xyz', 'kyriienko_nonlinear_encoding_circuit_n2_l1_local_z',
    'kyriienko_nonlinear_encoding_circuit_n2_l1_pauli_weight_le_2',
    'kyriienko_nonlinear_encoding_circuit_n4_l1_local_xyz', 'kyriienko_nonlinear_encoding_circuit_n4_l1_local_z',
    'kyriienko_nonlinear_encoding_circuit_n4_l1_pauli_weight_le_2',
    'hubregtsen_encoding_circuit_n4_l1_local_z'}
OPEN_BRACKETS = {'kyriienko_nonlinear_encoding_circuit_n2_l1_local_xyz': [4, 5],
                 'kyriienko_nonlinear_encoding_circuit_n4_l1_local_xyz': [10, 11],
                 'kyriienko_nonlinear_encoding_circuit_n4_l1_pauli_weight_le_2': [10, 11]}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, v):
    Path(p).write_text(json.dumps(v, indent=1, ensure_ascii=False, default=str) + '\n')


def manuscript_table():
    fam = {'YZ-CX': 'yz_cx_encoding_circuit', 'Hubregtsen': 'hubregtsen_encoding_circuit',
           'Kyriienko': 'kyriienko_nonlinear_encoding_circuit'}
    out, cur = {}, None
    text = TEX.read_text()
    body = text[text.index(r'\begin{tabular}'):text.index(r'\end{tabular}')]
    for line in body.splitlines():
        line = line.strip().rstrip('\\').strip()
        parts = [p.strip() for p in line.split('&')]
        if len(parts) != 7 or not parts[1].isdigit():
            continue
        if parts[0]:
            cur = fam[parts[0]]
        n, L, K = int(parts[1]), int(parts[2]), int(parts[3])
        for col, ro in zip(parts[4:], ('local_xyz', 'local_z', 'pauli_weight_le_2')):
            m = re.fullmatch(r'\[(\d+),(\d+)\]', col)
            if m:
                val = dict(kind='bracket', lower=int(m.group(1)), upper=int(m.group(2)))
            else:
                v = re.fullmatch(r'\$?(\d+)(\^\\dagger)?\$?', col)
                val = dict(kind='value', value=int(v.group(1)), dagger=bool(v.group(2)))
            out['%s_n%d_l%d_%s' % (cur, n, L, ro)] = dict(K=K, **val)
    assert len(out) == 30
    return out


def exact_jacobian(C, dC, weights):
    """Exact rational Jacobian rows (a<=b) of d_k G with an explicit common scale."""
    O, A, K = len(C), len(C[0]) if C else 0, len(dC)
    support = [a for a in range(A) if any(C[o][a] for o in range(O)) or any(dC[k][o][a] for k in range(K) for o in range(O))]
    den = 1
    for row in C:
        for x in row:
            den = fg.lcm(den, F(x).denominator)
    dden = 1
    for blk in dC:
        for row in blk:
            for x in row:
                dden = fg.lcm(dden, F(x).denominator)
    W = [F(w) for w in weights]
    wden = 1
    for w in W:
        wden = fg.lcm(wden, w.denominator)
    Ci = np.array([[int(C[o][a] * den) for a in support] for o in range(O)], dtype=object)
    Wi = np.array([int(w * wden) for w in W], dtype=object)
    CW = Ci * Wi[:, None]
    iu = np.triu_indices(len(support))
    cols = []
    for k in range(K):
        dCi = np.array([[int(dC[k][o][a] * dden) for a in support] for o in range(O)], dtype=object)
        Gk = dCi.T.dot(CW) + CW.T.dot(dCi)
        cols.append(Gk[iu])
    scale = F(1, den * dden * wden)
    rows = [[F(int(cols[k][r])) for k in range(K)] for r in range(len(iu[0]))]
    return rows, support, iu, scale


def interval_crosscheck(st, info, orig, eng_num):
    """Gram d^2 and gradient vs the copied original interval engine at the original witness point."""
    theta = orig['theta']
    x, xp = orig['pairs'][0]
    iv.dps = 30
    d, grad = ie.derivative(info, theta, x, xp)
    th = np.array([float(F(t)) for t in theta])
    J, C, dC, _ = eng_num.jacobian(th)
    xf, xpf = [float(F(v)) for v in x], [float(F(v)) for v in xp]
    dh = st.h(xf) - st.h(xpf)
    W = np.array([float(w) for w in st.w])
    G = C.T @ (W[:, None] * C)
    d_gram = float(dh @ G @ dh)
    g_gram = [float(dh @ (dC[k].T @ (W[:, None] * C) + C.T @ (W[:, None] * dC[k])) @ dh) for k in range(st.K)]
    return dict(theta=theta, x=x, xprime=xp, interval_dps=30,
                distance_abs_error=abs(d_gram - ie.mid(d)),
                gradient_max_abs_error=max(abs(a - ie.mid(b)) for a, b in zip(g_gram, grad)),
                agrees_within_1e_10=all(abs(a - ie.mid(b)) < 1e-10 for a, b in zip(g_gram + [d_gram], grad + [d])))


def numeric_crosschecks(st, eng, trials=32, seed=7):
    rng = np.random.default_rng(seed)
    W = np.array([float(w) for w in st.w])
    md = mg = 0.0
    for t in range(trials):
        theta = rng.uniform(-np.pi, np.pi, st.K)
        x, xp = rng.uniform(-0.99, 0.99, 2), rng.uniform(-0.99, 0.99, 2)
        C, dC = eng.coefficients(theta)
        md = max(md, abs(eng.distance_gram(C, x, xp) - eng.distance_direct(theta, x, xp)))
        phi = np.array([st.base_angles(x), st.base_angles(xp)])
        f, df = eng.simulate(theta, phi)
        dh = st.h(x) - st.h(xp)
        for k in range(st.K):
            direct = float(np.sum(2 * W * (f[:, 0] - f[:, 1]) * (df[:, k, 0] - df[:, k, 1])))
            gram = float(dh @ (dC[k].T @ (W[:, None] * C) + C.T @ (W[:, None] * dC[k])) @ dh)
            mg = max(mg, abs(direct - gram))
    return dict(trials=trials, max_abs_distance_error=md, max_abs_gradient_error=mg,
                passed=bool(md < 1e-10 and mg < 1e-10), tolerance=1e-10)


def analyze(name, index, sym_time=60.0, sym_terms=600000):
    t_all = time.perf_counter()
    info = json.loads((FROZEN / (name + '.json')).read_text())
    st = fg.Setting(info, name)
    rec = dict(id=name, parameter_count=st.K, n=st.n, observables=len(st.obs))
    rec.update(st.describe())
    timing = {}
    if not st.exact_supported:
        rec['status'] = 'NOT_FIXED_FREQUENCY_OR_UNSUPPORTED'
        return rec
    point = fg.pythagorean_point(st, SEED + index)
    rec['theta_star'] = dict(specification=point['specification'], float=point['theta_float'], seed=point['seed'],
                             zeta=[list(z) for z in point['zeta']])
    # ---------------- lower bound: exact rank of J_G(theta*) modulo primes ----
    lower = []
    for p in fg.PRIMES:
        t = time.perf_counter()
        eng = fg.LaurentEngine(st, point, 'mod', p).run()
        C, dC, real_ok = eng.coefficients_mod()
        J, support, iu = fg.gram_jacobian_mod(C, dC, st.w, p)
        r, prow, pcol = fg.rank_mod(J, p)
        minor = [[int(J[i, k]) for k in pcol] for i in prow]
        det = fg.det_mod(minor, p)
        assert (det != 0) == (r > 0) or r == 0
        lab = lambda a: '%s(%s)' % st.basis[int(support[a])]
        rows = [[lab(iu[0][i]), lab(iu[1][i])] for i in prow]
        active = {}
        for k in range(st.K):
            nzr = np.where(J[:, k] != 0)[0]
            if len(nzr):
                i = int(nzr[0])
                active[k] = dict(row=[lab(iu[0][i]), lab(iu[1][i])], value_mod_p=int(J[i, k]))
        lower.append(dict(prime=p, feature_coefficients_conjugate_symmetric=real_ok, A_support=int(len(support)),
                          A_full=len(st.basis), jacobian_rows=int(J.shape[0]), rank=r,
                          minor_rows_G_entries=rows, minor_columns=pcol, minor_det_mod_p=int(det),
                          active_witnesses=active, seconds=time.perf_counter() - t))
        if p == fg.PRIMES[0]:
            J0, prime0 = J, p
    timing['modular_rank_seconds'] = sum(x['seconds'] for x in lower)
    r_lower = max(x['rank'] for x in lower)
    rec['A'] = lower[0]['A_support']
    rec['lower_bound'] = dict(
        r_lower=r_lower, certificate='nonzero r x r minor of the exact rational Jacobian d upper(G)/d theta at '
        'theta*, certified by a nonzero determinant modulo a prime (all entries p-integral)', per_prime=lower)
    # ---------------- optional fully exact rational Jacobian --------------------
    if name in EXACT_RATIONAL:
        t = time.perf_counter()
        ee = fg.LaurentEngine(st, point, 'exact').run()
        Ce, dCe, labels = ee.coefficients_exact()
        rows, sup, iu_e, scale = exact_jacobian(Ce, dCe, st.w)
        re_, prow, pcol = fg.rank_exact(rows, st.K)
        det = fg.det_exact([[rows[i][k] for k in pcol] for i in prow]) * scale ** len(prow)
        # consistency of the exact coefficients with the modular engine
        p = fg.PRIMES[0]
        col = {lab: j for j, lab in enumerate(st.basis)}
        eng = fg.LaurentEngine(st, point, 'mod', p).run()
        Cm, dCm, _ = eng.coefficients_mod()
        mism = sum(int(fg.frac_mod(Ce[o][j], p) != Cm[o, col[l]]) for o in range(len(st.obs)) for j, l in enumerate(labels))
        mism += sum(int(fg.frac_mod(dCe[k][o][j], p) != dCm[k, o, col[l]]) for k in range(st.K)
                    for o in range(len(st.obs)) for j, l in enumerate(labels))
        Cn, dCn = fg.NumericEngine(st).coefficients(np.array(point['theta_float']))
        nerr = max([abs(float(Ce[o][j]) - Cn[o, col[l]]) for o in range(len(st.obs)) for j, l in enumerate(labels)] +
                   [abs(float(dCe[k][o][j]) - dCn[k, o, col[l]]) for k in range(st.K) for o in range(len(st.obs))
                    for j, l in enumerate(labels)])
        lab = lambda a: '%s(%s)' % labels[sup[a]]
        rec['lower_bound']['exact_rational'] = dict(
            rank=re_, minor_rows_G_entries=[[lab(iu_e[0][i]), lab(iu_e[1][i])] for i in prow], minor_columns=pcol,
            minor_determinant=str(det), minor_determinant_float=float(det),
            exact_vs_modular_coefficient_mismatches=mism, exact_vs_numeric_max_abs_error=nerr,
            seconds=time.perf_counter() - t)
        timing['exact_rational_seconds'] = time.perf_counter() - t
        r_lower = max(r_lower, re_)
        rec['lower_bound']['r_lower'] = r_lower
    # ---------------- upper bound U1: two-copy invariance null fields ----------
    t = time.perf_counter()
    work, inactive, log = fg.certified_reduction(st)
    vf = fg.virtual_fields(st, work)
    vecs, labels_f = fg.field_vectors(st, point, inactive, vf)
    frank = fg.rank_exact(vecs, st.K)[0] if vecs else 0
    jv = []
    for v, l in zip(vecs, labels_f):
        vm = np.array([fg.frac_mod(x, prime0) for x in v], dtype=np.int64)
        jv.append(dict(field=l, J_theta_star_times_v_is_zero_mod_p=bool((J0 @ vm % prime0 == 0).all())))
    timing['invariance_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    for entry in log:   # independent symbolic confirmation of every deletion step
        chk = fg.deletion_operator_check(st, entry, time_budget=sym_time, term_budget=sym_terms)
        entry.pop('_suffix')
        entry['fresh_angle_symbolic_check'] = chk
        if chk.get('completed') and not chk['fresh_angle_independent']:
            raise RuntimeError('Deletion step not confirmed symbolically %s gate %s' % (name, entry['source_gate']))
    timing['deletion_checks_seconds'] = time.perf_counter() - t
    U1 = st.K - frank
    upper = dict(U1=dict(bound=U1, inactive_symbols=inactive, deletion_log=log, virtual_fields=vf,
                         family_labels=labels_f, family_rank_at_theta_star=frank,
                         consistency_J_v_at_theta_star=jv,
                         argument=('Each field v(theta) satisfies J(theta) v(theta) = 0 for all theta: deletions keep '
                                   'K(x,x\';theta) = Tr[(rho(x)(x)rho(x\'))Omega] unchanged because the transported '
                                   'generator P commutes with Omega jointly (P(x)I + I(x)P) or separately (data '
                                   'angles); a virtual field is the Euler-block chart image of inserting exp(-i eps Q/2) '
                                   'whose transported generator is a joint symmetry. Exact rank of the family at '
                                   'theta* bounds the generic family rank from below, so r <= K - rank.')))
    # ---------------- upper bound U2 / symbolic Gram (budgeted) ----------------
    sym, blocks = {}, {}
    for lab_c, circ in (('original_circuit', st.gates), ('certified_reduced_circuit', work)):
        t = time.perf_counter()
        res = fg.symbolic_two_copy(st, circ, time_budget=sym_time, term_budget=sym_terms)
        blocks[lab_c] = res.get('block')
        out = {k: v for k, v in res.items() if k != 'block'}
        if res['completed']:
            span, nr, nc = fg.span_rank(res['block'])
            out.update(span_bound=span, gram_entries=nr, theta_frequencies=nc)
        out['seconds'] = time.perf_counter() - t
        sym[lab_c] = out
        timing['symbolic_%s_seconds' % lab_c] = out['seconds']
    upper['U2_symbolic_gram'] = sym
    # independent exact checks of every virtual field (symbolic, not numerical)
    t = time.perf_counter()
    for f in vf:
        f['suffix_operator_identity'] = fg.suffix_operator_check(st, work, f, time_budget=sym_time, term_budget=sym_terms)
        if blocks['original_circuit'] is not None:
            vt = fg.field_trig_poly(f)
            f['full_symbolic_gram_identity'] = fg.verify_field_on_symbolic_gram(blocks['original_circuit'], vt)
        if blocks['certified_reduced_circuit'] is not None:
            f['reduced_symbolic_gram_identity'] = fg.verify_field_on_symbolic_gram(blocks['certified_reduced_circuit'], fg.field_trig_poly(f))
        sc = f['suffix_operator_identity']
        if sc.get('completed') and not (sc['identity_holds'] and sc['perturbed_field_rejected']):
            raise RuntimeError('Virtual field check failed %s %s' % (name, f['symbols']))
        if f.get('full_symbolic_gram_identity') is False or f.get('reduced_symbolic_gram_identity') is False:
            raise RuntimeError('Virtual field symbolic Gram check failed %s' % name)
    timing['virtual_field_checks_seconds'] = time.perf_counter() - t
    cands = [('trivial K', st.K), ('U1 two-copy invariance fields', U1)]
    for lab_c, out in sym.items():
        if out.get('completed') and out.get('span_bound') is not None:
            cands.append(('U2 symbolic Gram span (%s)' % lab_c, out['span_bound']))
    r_upper, which = min((v, k) for k, v in cands)
    upper['r_upper'] = r_upper
    upper['attained_by'] = [k for k, v in cands if v == r_upper]
    rec['upper_bound'] = upper
    rec['r_lower'], rec['r_upper'] = r_lower, r_upper
    rec['status'] = 'CLOSED' if r_lower == r_upper else 'OPEN'
    assert r_lower <= r_upper, 'Certificate conflict'
    # ---------------- per-symbol decisions ----------------------------------
    active = set(lower[0]['active_witnesses']) | set(lower[1]['active_witnesses'])
    sym_inactive = set(sym['original_circuit']['inactive']) if sym['original_circuit'].get('completed') else set()
    status = []
    for k in range(st.K):
        certs = []
        if k in active:
            certs.append('ACTIVE: exact nonzero d_k G(theta*) (mod-p witness)')
        if k in sym_inactive:
            certs.append('INACTIVE: symbolic two-copy Gram expansion, d_k G == 0')
        if k in inactive:
            certs.append('INACTIVE: two-copy invariance deletion certificate')
        if k in active and (k in sym_inactive or k in inactive):
            raise RuntimeError('Conflicting per-symbol certificates %s %d' % (name, k))
        status.append(dict(symbol=k, verdict='ACTIVE' if k in active else 'INACTIVE' if certs else 'UNDECIDED',
                           certificates=certs))
    rec['per_symbol'] = status
    # ---------------- numerical diagnostics (NOT certificates) -----------------
    t = time.perf_counter()
    eng_num = fg.NumericEngine(st)
    rec['numerical_rank'] = fg.numerical_rank(st)
    rec['numeric_crosscheck'] = numeric_crosschecks(st, eng_num)
    timing['numeric_seconds'] = time.perf_counter() - t
    orig = json.loads((ORIG_RUN / (name + '.json')).read_text())
    t = time.perf_counter()
    rec['interval_engine_crosscheck'] = interval_crosscheck(st, info, orig, eng_num)
    timing['interval_crosscheck_seconds'] = time.perf_counter() - t
    rec['original_bounded_pass'] = dict(
        status_counts=orig['counts'], undecided=[k for k, s in enumerate(orig['status']) if s == 'UNDECIDED'],
        symbolic_failure=orig['symbolic_failure'], symbolic_seconds=orig['symbolic_seconds'],
        elapsed_seconds=orig['elapsed_seconds'], minor_lower=orig['minor']['lower'], r_upper=orig['r_upper'])
    timing['total_seconds'] = time.perf_counter() - t_all
    rec['timing'] = timing
    return rec


def worker(args):
    name, index = args
    try:
        return analyze(name, index)
    except Exception:
        return dict(id=name, error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    scope = json.loads((FROZEN / 'scope.json').read_text())
    names = [s['id'] for s in scope]
    if a.only:
        names = [n for n in names if n in a.only]
    out = HERE / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    table = manuscript_table()
    closure = {r['id']: r for r in json.loads((ORIG / 'structural_closure.json').read_text())['rows']}
    # provenance: hashes of everything read
    prov = dict(
        copied_files={p.name: dict(copy_sha256=sha(p), original_sha256=sha(ORIG / p.name),
                                   identical=sha(p) == sha(ORIG / p.name))
                      for p in sorted((HERE / 'copied_from_extension_v1').glob('*.py'))},
        frozen_inputs={}, scope_sha256=sha(FROZEN / 'scope.json'),
        manuscript_table_file=str(TEX.relative_to(JE)), manuscript_table_sha256=sha(TEX),
        structural_closure_sha256=sha(ORIG / 'structural_closure.json'),
        nonlinear_hubregtsen_sha256=sha(ORIG / 'nonlinear_hubregtsen.json'),
        original_results_manifest_sha256=sha(ORIG_RUN / 'manifest.json'),
        code_sha256={p.name: sha(p) for p in sorted(HERE.glob('*.py'))})
    for s in scope:
        p = FROZEN / (s['id'] + '.json')
        info = json.loads(p.read_text())
        prov['frozen_inputs'][s['id']] = dict(file_sha256=sha(p), matches_scope_file_sha256=sha(p) == s['file_sha256'],
                                              info_hash_matches_scope_source_sha256=ie.info_hash(info) == s['source_sha256'])
    assert all(v['matches_scope_file_sha256'] and v['info_hash_matches_scope_source_sha256'] for v in prov['frozen_inputs'].values())
    assert all(v['identical'] for v in prov['copied_files'].values())
    save(out / 'provenance.json', prov)
    save(out / 'environment.json', dict(python=platform.python_version(), numpy=np.__version__, sympy=sp.__version__,
                                        mpmath=mpmath.__version__, platform=platform.platform(),
                                        primes=list(fg.PRIMES), seed_base=SEED, workers=a.workers))
    t0 = time.perf_counter()
    jobs = [(nm, [s['id'] for s in scope].index(nm)) for nm in names]
    results = {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for rec in pool.map(worker, jobs):
            nm = rec['id']
            if 'error' not in rec:
                tv = table[nm]
                rec['manuscript'] = tv
                rec['structural_closure_bounds'] = [closure[nm]['rank_lower'], closure[nm]['rank_upper']]
                if tv['kind'] == 'value':
                    rec['consistency'] = 'AGREES' if rec['r_lower'] == rec['r_upper'] == tv['value'] else (
                        'CONSISTENT_NOT_CLOSED' if rec['r_lower'] <= tv['value'] <= rec['r_upper'] else 'DISAGREES')
                else:
                    closed = rec['r_lower'] == rec['r_upper']
                    inside = tv['lower'] <= rec['r_lower'] <= tv['upper']
                    rec['consistency'] = ('CLOSES_BRACKET' if closed and inside else 'DISAGREES' if closed
                                          else 'NOT_CLOSED' if rec['r_lower'] <= tv['upper'] and rec['r_upper'] >= tv['lower']
                                          else 'DISAGREES')
                rec['numerical_rank_matches_certified'] = rec['numerical_rank']['rank'] == rec['r_lower'] == rec['r_upper']
            results[nm] = rec
            save(out / (nm + '.json'), rec)
            print(nm, rec.get('r_lower'), rec.get('r_upper'), rec.get('status'), rec.get('consistency'),
                  'ERROR' if 'error' in rec else '', flush=True)
    recs = [results[n] for n in names]
    ok = [r for r in recs if 'error' not in r]
    budget_entries = []
    for r in ok:
        for k in r['original_bounded_pass']['undecided']:
            ps = r['per_symbol'][k]
            budget_entries.append(dict(setting=r['id'], symbol=k, gram_verdict=ps['verdict'], certificates=ps['certificates'],
                                       symbolic_gram_completed=r['upper_bound']['U2_symbolic_gram']['original_circuit'].get('completed'),
                                       symbolic_gram_seconds=r['upper_bound']['U2_symbolic_gram']['original_circuit']['seconds'],
                                       bounded_pass_symbolic_seconds=r['original_bounded_pass']['symbolic_seconds'],
                                       bounded_pass_failure=r['original_bounded_pass']['symbolic_failure']))
    summary = dict(
        settings=len(recs), errors=[r['id'] for r in recs if 'error' in r],
        fixed_frequency=[r['id'] for r in ok if r['fixed_frequency']],
        not_fixed_frequency=[r['id'] for r in ok if not r['fixed_frequency']],
        closed=[r['id'] for r in ok if r.get('status') == 'CLOSED'],
        open=[dict(id=r['id'], bracket=[r['r_lower'], r['r_upper']]) for r in ok if r.get('status') == 'OPEN'],
        open_brackets_from_manuscript={k: dict(manuscript=OPEN_BRACKETS[k], gram=[results[k]['r_lower'], results[k]['r_upper']] if k in results and 'error' not in results[k] else None,
                                               upper_attained_by=results[k]['upper_bound']['attained_by'] if k in results and 'error' not in results[k] else None,
                                               exact_minor=results[k]['lower_bound'].get('exact_rational') if k in results and 'error' not in results[k] else None)
                                       for k in OPEN_BRACKETS},
        consistency={r['id']: dict(gram=[r['r_lower'], r['r_upper']], manuscript=r['manuscript'], verdict=r['consistency'])
                     for r in ok},
        disagreements=[r['id'] for r in ok if r.get('consistency') == 'DISAGREES'],
        numerical_rank_mismatch=[r['id'] for r in ok if not r['numerical_rank_matches_certified']],
        numeric_crosscheck_failures=[r['id'] for r in ok if not r['numeric_crosscheck']['passed']],
        max_distance_error=max(r['numeric_crosscheck']['max_abs_distance_error'] for r in ok),
        max_gradient_error=max(r['numeric_crosscheck']['max_abs_gradient_error'] for r in ok),
        interval_crosscheck_max_gradient_error=max(r['interval_engine_crosscheck']['gradient_max_abs_error'] for r in ok),
        per_symbol_counts=dict(
            ACTIVE=sum(s['verdict'] == 'ACTIVE' for r in ok for s in r['per_symbol']),
            INACTIVE=sum(s['verdict'] == 'INACTIVE' for r in ok for s in r['per_symbol']),
            UNDECIDED=sum(s['verdict'] == 'UNDECIDED' for r in ok for s in r['per_symbol'])),
        budget_limited_entries=budget_entries,
        budget_limited_decided_by_symbolic_gram=sum(any('symbolic' in c for c in e['certificates']) for e in budget_entries),
        budget_limited_decided_by_invariance=sum(any('invariance' in c for c in e['certificates']) for e in budget_entries),
        symbolic_gram_completed_settings=[r['id'] for r in ok if r['upper_bound']['U2_symbolic_gram']['original_circuit'].get('completed')],
        wall_seconds=time.perf_counter() - t0)
    save(out / 'summary.json', summary)
    save(out / 'manifest.json', {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps({k: v for k, v in summary.items() if k not in ('consistency', 'budget_limited_entries', 'open_brackets_from_manuscript')}, indent=1))
    print('output', out)


if __name__ == '__main__':
    main()
