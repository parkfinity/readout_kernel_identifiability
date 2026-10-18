#!/usr/bin/env python3
"""Compare a fresh run of the two certificate computations against the stored reference results.

    python3 compare_results.py --run-dir runs/<UTC>            # what run_all.sh calls
    python3 compare_results.py --fresh-fourier DIR --fresh-kyriienko DIR [--out verdict.json]

Severity of every check
  strict  certificate content that must be bit-for-bit (or exactly, as rationals) identical:
          certified ranks and brackets, exact minor determinants (string AND Fraction equality),
          modular minors, minor rows/columns, per-symbol verdicts and their counts, deletion and
          virtual-field certificates, generated IR SHA256, Jacobian payload SHA256, code and
          frozen-input hashes, stage exit codes, bundle integrity.
  warn    numerical diagnostics that are NOT certificates (float SVD rank, float cross-checks)
          and environment/version mismatches.
  info    quantities that legitimately depend on wall-clock budgets or printing (whether the
          60 s / 600k-term symbolic two-copy expansion finished, certificate lists that
          include it, which bound attained r_upper, SymPy string forms of equal expressions).
Verdict: FAIL if any strict check fails, PASS_WITH_WARNINGS if only warnings, else PASS.
Exit code 0 for PASS / PASS_WITH_WARNINGS, 1 for FAIL.  Python >= 3.8; needs sympy only for
the symbolic-equality checks of virtual-field formulas.
"""
import argparse
import datetime
import glob
import gzip
import hashlib
import json
import os
import sys
import traceback
from fractions import Fraction

if hasattr(sys, 'set_int_max_str_digits'):   # CPython >= 3.8.14/3.9.14/3.10.7 cap int<->str at 4300 digits;
    sys.set_int_max_str_digits(0)            # the minor determinants have up to ~16k digits.

HERE = os.path.dirname(os.path.abspath(__file__))
REF_FOURIER = os.path.join(HERE, 'reference_results', 'fourier_gram', '20261002T044925Z')
REF_KYRIIENKO = os.path.join(HERE, 'reference_results', 'kyriienko_scaling', '20261003T121202Z')
EXPECTED_STAGES = ('fourier_unit_tests', 'kyriienko_unit_tests', 'fourier_gram_run', 'kyriienko_scaling_run')
FLOAT_TOL = 1e-10


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def load(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    with open(path, 'rb') as fh:
        return sha256_bytes(fh.read())


def compact(v):
    """Short, JSON-safe rendering of a possibly huge value for the report."""
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    if len(s) <= 160:
        return v
    return dict(length=len(s), sha256=sha256_bytes(s.encode())[:16], head=s[:60])


def frac(s):
    return Fraction(str(s))


def get(d, *path, default=None):
    for p in path:
        if isinstance(d, dict) and p in d:
            d = d[p]
        elif isinstance(d, list) and isinstance(p, int) and -len(d) <= p < len(d):
            d = d[p]
        else:
            return default
    return d


class Checks:
    def __init__(self):
        self.items = []

    def add(self, section, name, ok, severity='strict', ref=None, fresh=None, note=None):
        rec = dict(section=section, check=name, ok=bool(ok), severity=severity)
        if not ok:
            rec['reference'] = compact(ref)
            rec['fresh'] = compact(fresh)
        if note:
            rec['note'] = note
        self.items.append(rec)
        return bool(ok)

    def eq(self, section, name, ref, fresh, severity='strict', note=None):
        return self.add(section, name, ref == fresh, severity, ref, fresh, note)

    def keys(self, section, prefix, ref, fresh, keys, severity='strict'):
        for k in keys:
            self.eq(section, '%s.%s' % (prefix, k), get(ref, k), get(fresh, k), severity)

    def exception(self, section, name):
        self.add(section, name, False, 'strict', note='exception: ' + traceback.format_exc(limit=3)[-600:])

    def failures(self, severity):
        return [c for c in self.items if not c['ok'] and c['severity'] == severity]


def exact_rational_equal(ref, fresh):
    """(string_equal, value_equal, fresh_nonzero)."""
    try:
        return ref == fresh, frac(ref) == frac(fresh), frac(fresh) != 0
    except Exception:
        return ref == fresh, False, False


def sympy_equal(a, b):
    import sympy as sp
    return sp.expand(sp.sympify(a) - sp.sympify(b)) == 0


def verify_manifest(folder, manifest):
    """Return list of files whose SHA256 differs from the folder's own manifest."""
    if isinstance(manifest, dict) and 'files' in manifest:
        entries = [(f['path'], f['sha256']) for f in manifest['files']]
    else:
        entries = list(manifest.items())
    bad = []
    for rel, h in entries:
        p = os.path.join(folder, rel)
        if not os.path.exists(p) or sha256_file(p) != h:
            bad.append(rel)
    return bad, len(entries)


# ------------------------------------------------------------------------------------------
# Fourier-Gram closure (30 settings)
# ------------------------------------------------------------------------------------------
FG_TOP = ('parameter_count', 'n', 'observables', 'fixed_frequency', 'product_terms', 'constant_angle_terms',
          'exact_supported', 'data_bases', 'base_units_M', 'half_angle_step', 'max_half_angle_exponent',
          'basis_full_size', 'A', 'r_lower', 'r_upper', 'status', 'consistency', 'manuscript',
          'structural_closure_bounds', 'original_bounded_pass')
FG_PRIME = ('prime', 'feature_coefficients_conjugate_symmetric', 'A_support', 'A_full', 'jacobian_rows', 'rank',
            'minor_rows_G_entries', 'minor_columns', 'minor_det_mod_p', 'active_witnesses')
FG_DELETION = ('source_gate', 'rule', 'removed_symbols', 'removed_data', 'output_generator', 'sign')
FG_FIELD = ('qubit', 'source_gates', 'work_positions', 'block_paulis', 'symbols', 'coefficients', 'half_angle_m',
            'inserted_pauli', 'output_generator', 'output_sign', 'euler_identity_verified_by_groebner_reduction')
FG_U2 = ('inactive', 'span_bound', 'gram_entries', 'theta_frequencies', 'kernel_terms', 'gram_block_entries')


def verdict_counts(per_symbol):
    out = {}
    for s in per_symbol or []:
        out[s['verdict']] = out.get(s['verdict'], 0) + 1
    return out


def compare_fourier(ref_dir, fresh_dir, C, head, table):
    sec = 'fourier_gram'
    ref_sum = load(os.path.join(ref_dir, 'summary.json'))
    names = list(ref_sum['consistency'].keys())
    head.update(settings=len(names), certified_ranks_equal=0, exact_minor_determinants_compared=0,
                exact_minor_determinants_equal=0, modular_minor_determinants_compared=0,
                modular_minor_determinants_equal=0)
    if not fresh_dir or not os.path.isdir(fresh_dir):
        C.add(sec, 'fresh results folder exists', False, fresh=fresh_dir)
        return
    for name in names:
        s = '%s/%s' % (sec, name)
        try:
            ref = load(os.path.join(ref_dir, name + '.json'))
            fp = os.path.join(fresh_dir, name + '.json')
            if not C.add(s, 'fresh setting file present', os.path.exists(fp), fresh=fp):
                table[name] = dict(reference=[ref['r_lower'], ref['r_upper']], fresh=None, equal=False)
                continue
            new = load(fp)
            if not C.add(s, 'no error in fresh run', 'error' not in new, fresh=new.get('error')):
                table[name] = dict(reference=[ref['r_lower'], ref['r_upper']], fresh=None, equal=False)
                continue
            C.keys(s, 'setting', ref, new, FG_TOP)
            rank_eq = [ref['r_lower'], ref['r_upper']] == [new['r_lower'], new['r_upper']]
            head['certified_ranks_equal'] += int(rank_eq and new['status'] == ref['status'])
            C.keys(s, 'theta_star', ref['theta_star'], new['theta_star'], ('specification', 'zeta', 'seed'))
            C.eq(s, 'lower_bound.r_lower', ref['lower_bound']['r_lower'], new['lower_bound']['r_lower'])
            rp, npp = ref['lower_bound']['per_prime'], new['lower_bound']['per_prime']
            C.eq(s, 'lower_bound.per_prime.count', len(rp), len(npp))
            for i, (a, b) in enumerate(zip(rp, npp)):
                C.keys(s, 'lower_bound.per_prime[%d]' % i, a, b, FG_PRIME)
                head['modular_minor_determinants_compared'] += 1
                head['modular_minor_determinants_equal'] += int(a['minor_det_mod_p'] == b['minor_det_mod_p']
                                                                and a['minor_columns'] == b['minor_columns'])
            ex_eq = None
            if 'exact_rational' in ref['lower_bound']:
                a, b = ref['lower_bound']['exact_rational'], new['lower_bound'].get('exact_rational', {})
                C.keys(s, 'exact_rational', a, b, ('rank', 'minor_rows_G_entries', 'minor_columns'))
                st, val, nz = exact_rational_equal(a['minor_determinant'], b.get('minor_determinant'))
                C.add(s, 'exact_rational.minor_determinant string equal', st, ref=a['minor_determinant'],
                      fresh=b.get('minor_determinant'))
                C.add(s, 'exact_rational.minor_determinant Fraction equal', val)
                C.add(s, 'exact_rational.minor_determinant nonzero', nz)
                C.eq(s, 'exact_rational.exact_vs_modular_coefficient_mismatches (fresh) == 0', 0,
                     b.get('exact_vs_modular_coefficient_mismatches'))
                ex_eq = st and val and nz
                head['exact_minor_determinants_compared'] += 1
                head['exact_minor_determinants_equal'] += int(ex_eq)
            # ---- upper bound U1 (deterministic algebra) ----
            u1r, u1n = ref['upper_bound']['U1'], new['upper_bound']['U1']
            C.keys(s, 'U1', u1r, u1n, ('bound', 'inactive_symbols', 'family_labels', 'family_rank_at_theta_star'))
            C.eq(s, 'U1.deletion_log.count', len(u1r['deletion_log']), len(u1n['deletion_log']))
            for i, (a, b) in enumerate(zip(u1r['deletion_log'], u1n['deletion_log'])):
                C.keys(s, 'U1.deletion_log[%d]' % i, a, b, FG_DELETION)
                ca, cb = a.get('fresh_angle_symbolic_check', {}), b.get('fresh_angle_symbolic_check', {})
                if ca.get('completed') and cb.get('completed'):
                    C.eq(s, 'U1.deletion_log[%d].fresh_angle_independent' % i, ca.get('fresh_angle_independent'),
                         cb.get('fresh_angle_independent'))
                else:
                    C.eq(s, 'U1.deletion_log[%d].fresh_angle_check_completed (budget)' % i, ca.get('completed'),
                         cb.get('completed'), 'info')
            C.eq(s, 'U1.virtual_fields.count', len(u1r['virtual_fields']), len(u1n['virtual_fields']))
            for i, (a, b) in enumerate(zip(u1r['virtual_fields'], u1n['virtual_fields'])):
                C.keys(s, 'U1.virtual_fields[%d]' % i, a, b, FG_FIELD)
                for key in ('field_alpha_coordinates', 'cramer_determinant'):
                    ra, rb = a.get(key), b.get(key)
                    C.eq(s, 'U1.virtual_fields[%d].%s (SymPy string form)' % (i, key), ra, rb, 'info')
                    try:
                        if isinstance(ra, list):
                            same = len(ra) == len(rb) and all(sympy_equal(x, y) for x, y in zip(ra, rb))
                        else:
                            same = sympy_equal(ra, rb)
                        C.add(s, 'U1.virtual_fields[%d].%s symbolically equal' % (i, key), same, ref=ra, fresh=rb)
                    except Exception:
                        C.exception(s, 'U1.virtual_fields[%d].%s symbolic comparison' % (i, key))
                sa, sb = a.get('suffix_operator_identity', {}), b.get('suffix_operator_identity', {})
                if sa.get('completed') and sb.get('completed'):
                    C.keys(s, 'U1.virtual_fields[%d].suffix_operator_identity' % i, sa, sb,
                           ('identity_holds', 'perturbed_field_rejected'))
                else:
                    C.eq(s, 'U1.virtual_fields[%d].suffix_check_completed (budget)' % i, sa.get('completed'),
                         sb.get('completed'), 'info')
                for key in ('full_symbolic_gram_identity', 'reduced_symbolic_gram_identity'):
                    if a.get(key) is not None and b.get(key) is not None:
                        C.eq(s, 'U1.virtual_fields[%d].%s' % (i, key), a.get(key), b.get(key))
                    else:
                        C.eq(s, 'U1.virtual_fields[%d].%s present (budget)' % (i, key), a.get(key) is not None,
                             b.get(key) is not None, 'info')
            C.add(s, 'U1.J(theta*)v(theta*) == 0 mod p for every field (fresh)',
                  all(x['J_theta_star_times_v_is_zero_mod_p'] for x in u1n['consistency_J_v_at_theta_star']))
            C.eq(s, 'upper_bound.r_upper', ref['upper_bound']['r_upper'], new['upper_bound']['r_upper'])
            C.eq(s, 'upper_bound.attained_by (depends on symbolic budget)', ref['upper_bound']['attained_by'],
                 new['upper_bound']['attained_by'], 'info')
            for circ in ('original_circuit', 'certified_reduced_circuit'):
                a = ref['upper_bound']['U2_symbolic_gram'][circ]
                b = new['upper_bound']['U2_symbolic_gram'][circ]
                C.eq(s, 'U2.%s.completed (60 s / 600k-term budget)' % circ, a.get('completed'), b.get('completed'), 'info')
                if a.get('completed') and b.get('completed'):
                    C.keys(s, 'U2.%s' % circ, a, b, FG_U2)
            # ---- per-symbol verdicts ----
            C.eq(s, 'per_symbol.verdicts', [x['verdict'] for x in ref['per_symbol']],
                 [x['verdict'] for x in new['per_symbol']])
            C.eq(s, 'per_symbol.verdict_counts', verdict_counts(ref['per_symbol']), verdict_counts(new['per_symbol']))
            C.eq(s, 'per_symbol.certificate_lists (include budgeted symbolic Gram)',
                 [x['certificates'] for x in ref['per_symbol']], [x['certificates'] for x in new['per_symbol']], 'info')
            # ---- numerical diagnostics (not certificates) ----
            C.eq(s, 'numerical_rank.rank (diagnostic)', ref['numerical_rank']['rank'], new['numerical_rank']['rank'], 'warn')
            C.add(s, 'numerical_rank_matches_certified (diagnostic)', new.get('numerical_rank_matches_certified'), 'warn')
            C.add(s, 'numeric_crosscheck.passed (diagnostic, tol 1e-10)', get(new, 'numeric_crosscheck', 'passed'), 'warn')
            C.add(s, 'interval_engine_crosscheck.agrees_within_1e_10 (diagnostic)',
                  get(new, 'interval_engine_crosscheck', 'agrees_within_1e_10'), 'warn')
            table[name] = dict(reference=[ref['r_lower'], ref['r_upper']], fresh=[new['r_lower'], new['r_upper']],
                               equal=rank_eq, status=new['status'], exact_minor_det_equal=ex_eq,
                               verdict_counts=verdict_counts(new['per_symbol']))
        except Exception:
            C.exception(s, 'comparison')
    # ---- summary ----
    s = sec + '/summary'
    try:
        a, b = ref_sum, load(os.path.join(fresh_dir, 'summary.json'))
        C.keys(s, 'summary', a, b, ('settings', 'errors', 'fixed_frequency', 'not_fixed_frequency', 'closed', 'open',
                                    'disagreements', 'per_symbol_counts', 'budget_limited_decided_by_invariance'))
        C.eq(s, 'summary.consistency', a['consistency'], b['consistency'])
        for k, v in a['open_brackets_from_manuscript'].items():
            w = b['open_brackets_from_manuscript'].get(k, {})
            C.eq(s, 'open_bracket[%s].gram' % k, v['gram'], w.get('gram'))
            C.eq(s, 'open_bracket[%s].exact_minor.minor_determinant' % k, get(v, 'exact_minor', 'minor_determinant'),
                 get(w, 'exact_minor', 'minor_determinant'))
        proj = lambda e: [e['setting'], e['symbol'], e['gram_verdict']]  # noqa: E731
        C.eq(s, 'budget_limited_entries (setting, symbol, verdict)', [proj(e) for e in a['budget_limited_entries']],
             [proj(e) for e in b['budget_limited_entries']])
        for k in ('budget_limited_decided_by_symbolic_gram', 'symbolic_gram_completed_settings'):
            C.eq(s, 'summary.%s (budget dependent)' % k, a[k], b[k], 'info')
        C.eq(s, 'summary.numerical_rank_mismatch (diagnostic)', [], b['numerical_rank_mismatch'], 'warn')
        C.eq(s, 'summary.numeric_crosscheck_failures (diagnostic)', [], b['numeric_crosscheck_failures'], 'warn')
        head['per_symbol_counts'] = dict(reference=a['per_symbol_counts'], fresh=b['per_symbol_counts'])
        head['closed'] = len(b['closed'])
        head['budget_limited_inactive_by_invariance'] = b['budget_limited_decided_by_invariance']
        head['symbolic_gram_completed_settings'] = dict(reference=len(a['symbolic_gram_completed_settings']),
                                                        fresh=len(b['symbolic_gram_completed_settings']))
        head['wall_seconds_fresh'] = round(b.get('wall_seconds', 0), 1)
    except Exception:
        C.exception(s, 'summary comparison')
    # ---- provenance, environment, manifest ----
    s = sec + '/provenance'
    try:
        a, b = load(os.path.join(ref_dir, 'provenance.json')), load(os.path.join(fresh_dir, 'provenance.json'))
        C.keys(s, 'provenance', a, b, ('frozen_inputs', 'scope_sha256', 'manuscript_table_file', 'manuscript_table_sha256',
                                       'structural_closure_sha256', 'nonlinear_hubregtsen_sha256',
                                       'original_results_manifest_sha256', 'code_sha256', 'copied_files'))
        bad, n = verify_manifest(fresh_dir, load(os.path.join(fresh_dir, 'manifest.json')))
        C.add(s, 'fresh manifest.json hashes verify (%d files)' % n, not bad, fresh=bad)
        env = load(os.path.join(fresh_dir, 'environment.json'))
        head['fresh_environment'] = env
        C.keys(s, 'environment', load(os.path.join(ref_dir, 'environment.json')), env, ('primes', 'seed_base'))
    except Exception:
        C.exception(s, 'provenance comparison')


# ------------------------------------------------------------------------------------------
# Kyriienko scaling (n = 2, 4, 6, 8; 12 settings)
# ------------------------------------------------------------------------------------------
KY_TOP = ('setting', 'family', 'n', 'readout', 'K', 'conjecture_3n_minus_2', 'observables', 'ir_sha256', 'structure',
          'certified_r', 'status', 'interval', 'exact_consistency_at_theta_star')
KY_LOWER = ('value', 'method', 'theta_point', 'input_points', 'pairs', 'minor_rows_pair_indices', 'minor_rows_pairs',
            'minor_columns_theta', 'excluded_columns_theta', 'integer_minor_digits', 'residues', 'selection',
            'jacobian_file', 'jacobian_payload_sha256', 'sympy_domainmatrix_det_equal')
KY_MACHINE = ('deleted_coordinates', 'virtual_fields', 'family_rank_at_theta_star', 'upper',
              'exact_null_vectors_at_theta_star', 'logic')
KY_ROUTE = ('module', 'entry', 'constructor_kwargs', 'signature', 'get_params', 'initial_parameters', 'vendored_commit',
            'compile', 'native_operations', 'global_phase_ignored')
KY_TABLE = ('n', 'readout', 'K', 'conjecture_3n_minus_2', 'certified_lower', 'upper', 'certified_r', 'status',
            'minor_digits')


def compare_kyriienko(ref_dir, fresh_dir, C, head, table):
    sec = 'kyriienko_scaling'
    files = sorted(p for p in glob.glob(os.path.join(ref_dir, 'kyriienko_n*_*.json')))
    names = [os.path.basename(p)[:-5] for p in files]
    head.update(settings=len(names), certified_ranks_equal=0, exact_minor_determinants_equal=0,
                jacobian_payload_hashes_equal=0)
    if not fresh_dir or not os.path.isdir(fresh_dir):
        C.add(sec, 'fresh results folder exists', False, fresh=fresh_dir)
        return
    # ---- generated IR hashes ----
    s = sec + '/ir'
    try:
        ref_ir = {os.path.basename(p): sha256_file(p) for p in glob.glob(os.path.join(ref_dir, 'ir', '*.json'))}
        new_ir = {os.path.basename(p): sha256_file(p) for p in glob.glob(os.path.join(fresh_dir, 'ir', '*.json'))}
        eq = 0
        for fn in sorted(ref_ir):
            ok = C.eq(s, 'ir/%s sha256' % fn, ref_ir[fn], new_ir.get(fn))
            eq += int(ok)
        C.eq(s, 'ir file set', sorted(ref_ir), sorted(new_ir))
        head['ir_hashes'] = dict(compared=len(ref_ir), equal=eq,
                                 files={fn: dict(reference=ref_ir[fn], fresh=new_ir.get(fn)) for fn in sorted(ref_ir)})
    except Exception:
        C.exception(s, 'ir comparison')
    for name in names:
        s = '%s/%s' % (sec, name)
        try:
            ref = load(os.path.join(ref_dir, name + '.json'))
            fp = os.path.join(fresh_dir, name + '.json')
            if not C.add(s, 'fresh setting file present', os.path.exists(fp), fresh=fp):
                table[name] = dict(reference=ref['certified_r'], fresh=None, equal=False)
                continue
            new = load(fp)
            C.keys(s, 'setting', ref, new, KY_TOP)
            lr, ln = ref['certified_lower_bound'], new['certified_lower_bound']
            C.keys(s, 'lower', lr, ln, KY_LOWER)
            C.add(s, 'lower.sympy_domainmatrix_det_equal (fresh) is true', ln.get('sympy_domainmatrix_det_equal') is True)
            ok_det = True
            for key in ('integer_minor_determinant', 'row_scale_product', 'rational_minor_determinant'):
                st, val, nz = exact_rational_equal(lr[key], ln.get(key))
                C.add(s, 'lower.%s string equal' % key, st, ref=lr[key], fresh=ln.get(key))
                C.add(s, 'lower.%s exact value equal' % key, val)
                ok_det = ok_det and st and val
            nz_needed = lr['value'] > 0
            C.add(s, 'lower.integer_minor_determinant nonzero (order > 0)',
                  (frac(ln['integer_minor_determinant']) != 0) if nz_needed else True)
            proj = lambda t: [t.get('theta_attempt'), t.get('points'), t.get('pairs'), t.get('rank_mod_p'), t.get('field_upper')]  # noqa: E731
            C.eq(s, 'lower.attempts (attempt, points, pairs, rank_mod_p, field_upper)',
                 [proj(t) for t in lr['attempts']], [proj(t) for t in ln['attempts']])
            # stored integer Jacobian payload
            jp = os.path.join(fresh_dir, ln['jacobian_file'])
            if C.add(s, 'fresh jacobian file present', os.path.exists(jp), fresh=jp):
                with open(jp, 'rb') as fh:
                    payload_sha = sha256_bytes(gzip.decompress(fh.read()))
                C.eq(s, 'fresh jacobian payload sha256 == its record', ln['jacobian_payload_sha256'], payload_sha)
                same_payload = C.eq(s, 'jacobian payload sha256 == reference', lr['jacobian_payload_sha256'], payload_sha)
                head['jacobian_payload_hashes_equal'] += int(same_payload)
            ur, un = ref['upper_bound'], new['upper_bound']
            C.keys(s, 'upper', ur, un, ('value', 'manuscript_value', 'manuscript_reference'))
            C.keys(s, 'upper.machine_certificate', ur['machine_certificate'], un['machine_certificate'], KY_MACHINE)
            C.eq(s, 'cross_checks.exact_d2_first_pairs', get(ref, 'cross_checks', 'exact_d2_first_pairs'),
                 get(new, 'cross_checks', 'exact_d2_first_pairs'))
            counts_r = dict(deleted_inactive=len(ur['machine_certificate']['deleted_coordinates']),
                            virtual_fields=len(ur['machine_certificate']['virtual_fields']),
                            minor_columns=len(lr['minor_columns_theta']))
            counts_n = dict(deleted_inactive=len(un['machine_certificate']['deleted_coordinates']),
                            virtual_fields=len(un['machine_certificate']['virtual_fields']),
                            minor_columns=len(ln['minor_columns_theta']))
            C.eq(s, 'per_symbol counts (deleted / virtual fields / minor columns)', counts_r, counts_n)
            C.keys(s, 'source_route', ref['source_route'], new['source_route'], KY_ROUTE)
            C.eq(s, 'source_route.qiskit', '1.2.4', get(new, 'source_route', 'qiskit'), 'warn')
            # diagnostics
            num = new.get('numerical_not_certified') or {}
            C.eq(s, 'numerical SVD rank == certified (diagnostic)', new['certified_r'], num.get('rank'), 'warn')
            cc = new.get('cross_checks', {})
            for key in ('features_exact_vs_float_ir_engine_max_abs', 'features_exact_vs_dense_source_statevector_max_abs',
                        'features_exact_vs_qiskit_source_statevector_max_abs', 'gradient_exact_vs_float_tangent_max_abs'):
                v = cc.get(key)
                C.add(s, 'cross_checks.%s < 1e-10 (diagnostic)' % key, v is not None and v < FLOAT_TOL, 'warn', fresh=v)
            r_eq = ref['certified_r'] == new['certified_r'] and new['status'] == ref['status']
            head['certified_ranks_equal'] += int(r_eq)
            head['exact_minor_determinants_equal'] += int(ok_det)
            table[name] = dict(reference=ref['certified_r'], fresh=new['certified_r'], equal=r_eq, status=new['status'],
                               interval=new['interval'], exact_minor_det_equal=ok_det,
                               ir_sha256_equal=ref['ir_sha256'] == new['ir_sha256'], per_symbol_counts=counts_n)
        except Exception:
            C.exception(s, 'comparison')
    s = sec + '/global'
    try:
        a, b = load(os.path.join(ref_dir, 'validation_gate.json')), load(os.path.join(fresh_dir, 'validation_gate.json'))
        C.add(s, 'validation_gate.passed (fresh)', b.get('passed') is True)
        C.add(s, 'validation_gate.secondary_all_match (fresh)', b.get('secondary_all_match') is True)
        C.eq(s, 'validation_gate.ir (generated vs frozen n=2,4 IR)', a['ir'], b['ir'])
        C.eq(s, 'validation_gate.ranks', a['ranks'], b['ranks'])
        C.eq(s, 'euler_identity.json', load(os.path.join(ref_dir, 'euler_identity.json')),
             load(os.path.join(fresh_dir, 'euler_identity.json')))
        sa, sb = load(os.path.join(ref_dir, 'summary.json')), load(os.path.join(fresh_dir, 'summary.json'))
        C.keys(s, 'summary', sa, sb, ('family', 'sizes', 'validation_gate_passed'))
        C.eq(s, 'summary.table (certified columns)', [[r[k] for k in KY_TABLE] for r in sa['table']],
             [[r.get(k) for k in KY_TABLE] for r in sb['table']])
        C.eq(s, 'summary.table numerical ranks (diagnostic)', [r['numerical_rank_not_certified'] for r in sa['table']],
             [r.get('numerical_rank_not_certified') for r in sb['table']], 'warn')
        ar, an = load(os.path.join(ref_dir, 'audit_replay.json')), load(os.path.join(fresh_dir, 'audit_replay.json'))
        C.add(s, 'audit_replay.passed (fresh, determinants recomputed from stored Jacobians)', an.get('passed') is True)
        C.eq(s, 'audit_replay.settings', ar['settings'], an['settings'])
        pa, pb = load(os.path.join(ref_dir, 'provenance.json')), load(os.path.join(fresh_dir, 'provenance.json'))
        C.keys(s, 'provenance', pa, pb, ('copied_core', 'copied_core_identical', 'code', 'frozen_kyriienko_inputs',
                                        'manuscript_files'))
        C.add(s, 'provenance.copied_core_identical (fresh)', pb.get('copied_core_identical') is True)
        bad, n = verify_manifest(fresh_dir, load(os.path.join(fresh_dir, 'manifest.json')))
        C.add(s, 'fresh manifest.json hashes verify (%d files)' % n, not bad, fresh=bad)
        env = load(os.path.join(fresh_dir, 'environment.json'))
        head['fresh_environment'] = {k: env.get(k) for k in ('python', 'platform', 'numpy', 'sympy', 'qiskit', 'scipy')}
        head['validation_gate_passed'] = b.get('passed')
        head['audit_replay_passed'] = an.get('passed')
        head['wall_seconds_fresh'] = round(sb.get('wall_seconds', 0), 1)
        head['peak_rss_mb_fresh'] = round(sb.get('peak_rss_mb_process', 0), 1)
    except Exception:
        C.exception(s, 'global comparison')


# ------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run-dir', help='run folder written by run_all.sh (implies the paths below)')
    ap.add_argument('--fresh-fourier')
    ap.add_argument('--fresh-kyriienko')
    ap.add_argument('--ref-fourier', default=REF_FOURIER)
    ap.add_argument('--ref-kyriienko', default=REF_KYRIIENKO)
    ap.add_argument('--stage-status')
    ap.add_argument('--env-check')
    ap.add_argument('--integrity')
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    if a.run_dir:
        rd = os.path.abspath(a.run_dir)
        a.fresh_fourier = a.fresh_fourier or os.path.join(rd, 'fourier_gram')
        a.fresh_kyriienko = a.fresh_kyriienko or os.path.join(rd, 'kyriienko_scaling')
        a.stage_status = a.stage_status or os.path.join(rd, 'stage_status.json')
        a.env_check = a.env_check or os.path.join(rd, 'environment_check.json')
        a.integrity = a.integrity or os.path.join(rd, 'bundle_integrity.json')
        a.out = a.out or os.path.join(rd, 'comparison_verdict.json')
    C = Checks()
    heads = dict(fourier_gram={}, kyriienko_scaling={})
    tables = dict(fourier_gram={}, kyriienko_scaling={})
    run_info = {}
    if a.stage_status:
        if C.add('run', 'stage_status.json present', os.path.exists(a.stage_status), fresh=a.stage_status):
            st = {x['name']: x for x in load(a.stage_status)['stages']}
            run_info['stages'] = list(st.values())
            for name in EXPECTED_STAGES:
                C.eq('run', 'stage %s exit code' % name, 0, get(st, name, 'exit_code'))
    if a.integrity:
        if C.add('run', 'bundle_integrity.json present', os.path.exists(a.integrity), fresh=a.integrity):
            rep = load(a.integrity)
            C.add('run', 'bundle files match BUNDLE_MANIFEST.json', rep.get('passed') is True,
                  fresh=dict(missing=rep.get('missing'), changed=rep.get('changed')))
    if a.env_check and os.path.exists(a.env_check):
        env = load(a.env_check)
        run_info['environment'] = dict(versions=env.get('versions'), platform=env.get('platform'),
                                       cpu_count=env.get('cpu_count'), warnings=env.get('warnings'),
                                       gcd_shim_used=env.get('gcd_shim_will_be_used_for_kyriienko'))
        for w in env.get('warnings', []):
            C.add('environment', 'version: ' + w, False, 'warn')
        for m in env.get('missing_required', []):
            C.add('environment', 'required module missing: ' + m, False, 'strict')
    try:
        compare_fourier(os.path.abspath(a.ref_fourier), a.fresh_fourier and os.path.abspath(a.fresh_fourier), C,
                        heads['fourier_gram'], tables['fourier_gram'])
    except Exception:
        C.exception('fourier_gram', 'top-level')
    try:
        compare_kyriienko(os.path.abspath(a.ref_kyriienko), a.fresh_kyriienko and os.path.abspath(a.fresh_kyriienko), C,
                          heads['kyriienko_scaling'], tables['kyriienko_scaling'])
    except Exception:
        C.exception('kyriienko_scaling', 'top-level')
    strict, warn = C.failures('strict'), C.failures('warn')
    info = C.failures('info')
    verdict = 'FAIL' if strict else ('PASS_WITH_WARNINGS' if warn else 'PASS')
    meaning = {
        'PASS': 'Every certificate quantity of the fresh run equals the stored reference exactly; no warnings.',
        'PASS_WITH_WARNINGS': 'Every certificate quantity equals the reference exactly; only non-certificate '
                              'diagnostics or environment versions differ (see warnings).',
        'FAIL': 'At least one certificate quantity, IR hash, stage exit code or bundle file differs (see strict_failures).'}
    sections = {}
    for c in C.items:
        top = c['section'].split('/')[0]
        d = sections.setdefault(top, dict(checks=0, strict_failures=0, warnings=0, info_differences=0))
        d['checks'] += 1
        if not c['ok']:
            d[{'strict': 'strict_failures', 'warn': 'warnings', 'info': 'info_differences'}[c['severity']]] += 1
    out = dict(
        verdict=verdict, meaning=meaning[verdict], generated_utc=utc_now(),
        inputs=dict(reference_fourier=a.ref_fourier, reference_kyriienko=a.ref_kyriienko, fresh_fourier=a.fresh_fourier,
                    fresh_kyriienko=a.fresh_kyriienko),
        counts=dict(checks=len(C.items), strict_checks=sum(c['severity'] == 'strict' for c in C.items),
                    strict_failures=len(strict), warnings=len(warn), info_differences=len(info)),
        sections=sections, headline=heads, run=run_info, per_setting=tables,
        strict_failures=strict, warnings=warn, info_differences=info,
        severity_policy=__doc__.split('Severity of every check')[1].split('Exit code')[0].strip())
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
            fh.write('\n')
    fg, ky = heads['fourier_gram'], heads['kyriienko_scaling']
    print('=' * 78)
    print('VERDICT: %s  (%d checks, %d strict; strict failures %d, warnings %d, info differences %d)' % (
        verdict, out['counts']['checks'], out['counts']['strict_checks'], len(strict), len(warn), len(info)))
    print('Fourier-Gram: certified ranks equal %s/%s, exact minor dets equal %s/%s, modular minors equal %s/%s, '
          'per-symbol counts fresh %s' % (fg.get('certified_ranks_equal'), fg.get('settings'),
                                         fg.get('exact_minor_determinants_equal'), fg.get('exact_minor_determinants_compared'),
                                         fg.get('modular_minor_determinants_equal'), fg.get('modular_minor_determinants_compared'),
                                         get(fg, 'per_symbol_counts', 'fresh')))
    print('Kyriienko:    certified ranks equal %s/%s, exact minor dets equal %s/%s, IR hashes equal %s/%s, '
          'Jacobian payloads equal %s/%s, gate %s, audit %s' % (
              ky.get('certified_ranks_equal'), ky.get('settings'), ky.get('exact_minor_determinants_equal'),
              ky.get('settings'), get(ky, 'ir_hashes', 'equal'), get(ky, 'ir_hashes', 'compared'),
              ky.get('jacobian_payload_hashes_equal'), ky.get('settings'), ky.get('validation_gate_passed'),
              ky.get('audit_replay_passed')))
    for c in strict[:25]:
        print('  STRICT FAIL  %s :: %s' % (c['section'], c['check']))
    for c in warn[:25]:
        print('  WARN         %s :: %s' % (c['section'], c['check']))
    if info:
        print('  (%d info-level differences, e.g. symbolic-budget completion; see the JSON)' % len(info))
    if a.out:
        print('verdict JSON: %s' % a.out)
    print('=' * 78)
    return 1 if verdict == 'FAIL' else 0


if __name__ == '__main__':
    sys.exit(main())
