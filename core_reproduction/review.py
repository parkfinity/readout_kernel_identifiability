#!/usr/bin/env python3
"""Portable reviewer entry point; scientific modules are retained unchanged.

References are immutable. Each run gets a private working copy and full logs.
No cloud credentials, QPU access, dataset download or public upload is used.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
PILOT = 'pilot_20261001T161006_106481Z'
GRAM = '20261002T044925Z'
TIMING_SHA256 = 'f8185e3287a9bbb602062f64cb483892c586b6433d981be85de6063487a5e871'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text())

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

def integrity():
    m = read(ROOT/'MANIFEST.json')
    for rel, expected in m['files'].items():
        p = ROOT/rel
        if not p.is_file() or sha(p) != expected:
            raise RuntimeError('Package hash mismatch: '+rel)
    return {'passed': True, 'files_checked': len(m['files']),
            'scope': 'Accidental-corruption detection, not independent scientific verification.'}

def unpack(archive, dest):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive names')
        for n in names:
            p = PurePosixPath(n)
            if p.is_absolute() or '..' in p.parts or '\\' in n:
                raise ValueError('Unsafe archive member: '+n)
        if z.testzip() is not None:
            raise ValueError('Archive CRC failure')
        z.extractall(dest)
    return next(dest.rglob('summary.json')).parent

def census(pilot, gram, output, compare=True):
    rows = []
    for p in sorted(pilot.glob('*.json')):
        r = read(p)
        if 'minor' not in r or 'exact' not in r:
            continue
        lo, hi = r['minor']['lower'], r['exact']['r_upper']
        if lo != hi:
            raise ValueError('Open pilot rank: '+r['id'])
        if compare:
            old = read(ROOT/'core/journal_extension/certified_dimension_pilot_v1/results'/PILOT/p.name)
            if (r['status'], lo, hi, r['exact']) != (old['status'], old['minor']['lower'], old['exact']['r_upper'], old['exact']):
                raise ValueError('Pilot reference mismatch: '+r['id'])
        rows.append([r['id'], '24-setting exact/interval', r['parameter_count'],
                     lo, hi, r['counts'].get('ACTIVE_CERTIFIED', 0),
                     r['counts'].get('EXACT_INACTIVE', 0)])
    if len(rows) != 24:
        raise ValueError('Expected 24 pilot rows')
    for p in sorted(gram.glob('*.json')):
        r = read(p)
        if 'r_lower' not in r:
            continue
        if r['r_lower'] != r['r_upper'] or r['status'] != 'CLOSED':
            raise ValueError('Open Gram rank: '+r['id'])
        if compare:
            old = read(ROOT/'core/reference_results/fourier_gram'/GRAM/p.name)
            if (r['r_lower'], r['r_upper'], r['per_symbol']) != (old['r_lower'], old['r_upper'], old['per_symbol']):
                # Some per-symbol fields are diagnostic; the core comparator separately
                # checks the detailed certificates. Only verdicts are census invariants.
                verdicts = lambda x: [v['verdict'] for v in x['per_symbol']]
                if (r['r_lower'], r['r_upper'], verdicts(r)) != (old['r_lower'], old['r_upper'], verdicts(old)):
                    raise ValueError('Gram census mismatch: '+r['id'])
        labels = [v['verdict'] for v in r['per_symbol']]
        rows.append([r['id'], '30-setting Fourier-Gram', r['parameter_count'],
                     r['r_lower'], r['r_upper'], labels.count('ACTIVE'), labels.count('INACTIVE')])
    if len(rows) != 54:
        raise ValueError('Expected 54 census rows')
    with (output/'census_54.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['setting', 'certificate_route', 'source_symbols', 'rank_lower', 'rank_upper', 'active', 'inactive'])
        w.writerows(rows)
    save(output/'census_verdict.json', {'passed': True, 'settings': 54,
         'open_rank_intervals': 0, 'generated_from': [str(pilot), str(gram)],
         'note': 'A table extraction alone is not a certificate replay; see stage log.'})

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['integrity', 'smoke', 'audit', 'full', 'resources', 'diagnostics',
                                    'timing-smoke', 'timing-full', 'timing-archive'])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', type=Path, default=ROOT/'runs')
    ap.add_argument('--archive', type=Path, help='Optional original timing-evidence ZIP')
    args = ap.parse_args()
    if sys.flags.optimize:
        ap.error('Do not use python -O or PYTHONOPTIMIZE: scientific checks use assertions.')
    if not 1 <= args.workers <= 8:
        ap.error('--workers must be 1..8')
    # Disable the integer-string guard only for exact determinant serialization.
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(0)
    checked = integrity()
    if args.mode == 'integrity':
        print(json.dumps(checked, indent=2)); return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = args.out.resolve()/(args.mode+'_'+stamp)
    out.mkdir(parents=True, exist_ok=False)
    save(out/'integrity.json', checked)
    packages = {}
    for name in ('numpy', 'sympy', 'mpmath', 'qiskit', 'scipy', 'scikit-learn'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = 'not installed'
    save(out/'environment.json', {'python': sys.version, 'platform': platform.platform(),
                                 'packages': packages, 'workers': args.workers})
    work = out/'work'
    shutil.copytree(ROOT/'core', work)
    je = work/'journal_extension'
    env = os.environ.copy()
    env.update({k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS',
                               'PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED')})
    env.update(PYTHONHASHSEED='0', PYTHONINTMAXSTRDIGITS='0', WORKERS=str(args.workers), PY=sys.executable)
    env.pop('PYTHONOPTIMIZE', None)
    statuses = []
    def stage(name, cwd, command):
        print('[RUN]', name, flush=True)
        start = time.monotonic()
        log = out/'logs'/(name+'.log'); log.parent.mkdir(exist_ok=True)
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while True:
                    try:
                        code = process.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        print('[WORKING]', name, '%.0fs; log:' % (time.monotonic()-start), log, flush=True)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                raise
        statuses.append({'stage': name, 'exit_code': code,
                         'elapsed_seconds': time.monotonic()-start, 'log': str(log.relative_to(out))})
        save(out/'stages.json', statuses)
        if code:
            print(log.read_text()[-8000:], flush=True)
            raise RuntimeError(name+' failed; see '+str(log))
        print('[PASS]', name, '%.1fs' % statuses[-1]['elapsed_seconds'], flush=True)
    def py(name, cwd, *command):
        stage(name, cwd, [sys.executable, *map(str, command)])
    pilot = je/'certified_dimension_pilot_v1'
    ext = je/'certified_dimension_extension_v1'
    fg = je/'fourier_gram_identifiability_v1/experiment'
    scaling = je/'kyriienko_scaling_v1'
    checks = je/'manuscript_checks'
    started = time.monotonic()
    try:
        if args.mode in ('smoke', 'audit', 'full'):
            for name, folder, modules in (
                ('pilot_tests', pilot, ['test_engine']),
                ('extension_tests', ext, ['test_engine', 'test_extension']),
                ('gram_tests', fg, ['test_fourier_gram']),
                ('scaling_tests', scaling, ['test_kyriienko_scaling']),
                ('normalizer_tests', je/'global_pauli_reparameterization_v1', ['test_normalizer']),
                ('survey_tests', je/'cross_family_resource_v1', ['test_survey'])):
                # Same compatibility shim as the original bundle on Python 3.8.
                py(name, folder, work/'tools/repro_helper.py', 'pyexec', '--', '-m', 'unittest', '-v', *modules)
            py('endpoint_exact_minor', checks, 'check_n2_endpoint.py')
            py('representative_exact_r4', checks, 'verify_exact_claims.py')
        if args.mode in ('audit', 'full'):
            if args.mode == 'full':
                before = set((pilot/'results').iterdir())
                py('pilot_fresh_24', pilot, 'run_pilot.py')
                new = set((pilot/'results').iterdir())-before
                if len(new) != 1: raise RuntimeError('Missing or ambiguous fresh pilot output')
                prun = new.pop()
            else:
                prun = pilot/'results'/PILOT
            py('pilot_certificate_replay', pilot, 'run_pilot.py', '--audit', prun,
               '--audit-output', out/'pilot_replay.json')
            py('pilot_source_crosscheck', pilot, 'source_crosscheck.py', prun,
               '--output', out/'pilot_source_check.json')
            py('nonlinear_exact', ext, 'nonlinear_hubregtsen.py', '--output', out/'nonlinear_exact.json')
            py('nonlinear_source_check', ext, 'check_nonlinear.py', '--proof', out/'nonlinear_exact.json',
               '--output', out/'nonlinear_source_check.json')
            py('legacy_csv_and_qfi_check', checks, 'review_checks.py')
            py('bounded_analytic_checks', checks, 'check_extensions.py')
            if args.mode == 'full':
                stage('gram_30_and_scaling_12_fresh', work, ['bash', 'run_all.sh'])
                runs = list((work/'runs').glob('*/comparison_verdict.json'))
                if len(runs) != 1: raise RuntimeError('Missing or ambiguous core comparison')
                core_run = runs[0].parent
                shutil.copy2(runs[0], out/'core_comparison_verdict.json')
                cv = read(runs[0])
                print('[CORE VERDICT]', cv['verdict'], cv['counts'], flush=True)
                grun = core_run/'fourier_gram'
            else:
                grun = work/'reference_results/fourier_gram'/GRAM
            census(prun, grun, out)
            py('extension_resource_accounting_30', ext, ROOT/'extension_resource_check.py',
               '--module', ext, '--gram', grun, '--output', out/'extension_resource_check.json')
        if args.mode in ('resources', 'full'):
            survey = je/'cross_family_resource_v1'
            global_code = je/'global_pauli_reparameterization_v1'
            py('survey_fresh_102', survey, 'survey.py', '--output', out/'resource_survey')
            py('survey_array_audit', survey, 'audit_results.py', out/'resource_survey')
            py('global_normalization_fresh_12', global_code, 'experiment.py', '--output', out/'global_normalization')
            sr = unpack(ROOT/'evidence/cross_family_server_results.zip', out/'survey_reference')
            gr = unpack(ROOT/'evidence/global_pauli_server_results.zip', out/'global_reference')
            now, old = read(out/'resource_survey/summary.json'), read(sr/'summary.json')
            for key in ('nominal_slots', 'statuses', 'execution_resource_positive_slots', 'additional_symbol_slots'):
                if now[key] != old[key]: raise RuntimeError('Resource census mismatch: '+key)
            nowg, oldg = read(out/'global_normalization/summary.json'), read(gr/'summary.json')
            a = {r['id']: r['metrics'] for r in nowg['rows']}
            b = {r['id']: r['metrics'] for r in oldg['rows']}
            if a != b: raise RuntimeError('Global normalization resource-count mismatch')
            save(out/'resource_comparison.json', {'passed': True, 'distinct_survey_slots': 102,
                'nominal_slots': 120, 'global_settings': 12, 'incremental_resource_positive_slots': now['execution_resource_positive_slots'],
                'scope': 'Fresh ideal-simulator validation and schedule counts, not QPU/shots or timing advantage.'})
        if args.mode in ('timing-smoke', 'timing-full'):
            folder = je/'interior_incremental_v1'
            py(args.mode, folder, 'runner.py', 'smoke' if args.mode == 'timing-smoke' else 'full')
        if args.mode == 'diagnostics':
            py('sampled_direct_summand_diagnostic', checks, 'check_ds_constant.py')
            shutil.copy2(checks/'analysis/ds_constant_check.json', out/'ds_constant_check.json')
        if args.mode == 'timing-archive':
            if not args.archive or sha(args.archive.resolve()) != TIMING_SHA256:
                raise ValueError('Supply the byte-identical optional timing companion with --archive')
            py('timing_saved_pair_reaggregation', je/'full_review_20261001', 'audit_full.py', args.archive.resolve())
            shutil.copy2(je/'full_review_20261001/full_analysis.json', out/'timing_reaggregation.json')
        core_status = read(out/'core_comparison_verdict.json') if (out/'core_comparison_verdict.json').exists() else None
        save(out/'verdict.json', {'passed': True, 'mode': args.mode, 'elapsed_seconds': time.monotonic()-started,
            'core_verdict': core_status['verdict'] if core_status else None,
            'core_warnings': core_status['counts']['warnings'] if core_status else 0,
            'stages': len(statuses), 'scope': 'See README.txt and CLAIM_MAP.txt; audit and smoke are NOT full reruns.'})
    except Exception as e:
        save(out/'verdict.json', {'passed': False, 'mode': args.mode, 'error': str(e), 'stages': statuses})
        raise
    print('PASS — results:', out, flush=True)

if __name__ == '__main__':
    main()
