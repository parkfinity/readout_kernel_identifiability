#!/usr/bin/env python3
"""Helper for reproduction_bundle_v1 (Python >= 3.8, standard library only at import time).

Subcommands
  env        print python / numpy / sympy / qiskit (+ mpmath, scipy, rustworkx, symengine) versions,
             WARN (never fail) when they differ from the server pins, write a JSON record.
  integrity  verify the SHA256 of every bundle file against BUNDLE_MANIFEST.json.
  run        run one stage as a subprocess: tee its output to a log, record exit code, wall time
             and peak RSS (largest single process of the stage) in a status JSON.
  pyexec     run a Python script or "-m module" in this interpreter, first installing the
             Python 3.8 compatibility shim for math.gcd (see below).
  build-manifest   (bundle maintainer only) write BUNDLE_MANIFEST.json.

Python 3.8 compatibility shim.  kyriienko_scaling.py calls math.gcd with three arguments and
with *row (variadic gcd exists only since Python 3.9).  On Python 3.8 "pyexec" replaces
math.gcd by a variadic wrapper with exactly the 3.9 semantics (gcd() = 0, gcd(a) = |a|,
gcd(a, b, ...) = fold of the 2-argument gcd).  The bundled source files stay byte-identical to
the originals (their SHA256 is checked by compare_results.py).  On Python >= 3.9 the shim is not
installed unless REPRO_FORCE_GCD_SHIM=1 (used only to test the shim itself).
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import resource
import runpy
import subprocess
import sys
import time

EXPECTED = {'python': '3.8.10', 'numpy': '1.24.4', 'sympy': '1.13.3', 'qiskit': '1.2.4'}
INFORMATIONAL = {'mpmath': '1.3.0', 'scipy': '1.10.1'}
REFERENCE_RUNS = {
    'fourier_gram 20261002T044925Z': 'Python 3.9.6, NumPy 1.26.4, SymPy 1.14.0, mpmath 1.3.0 (no Qiskit used), macOS arm64',
    'kyriienko_scaling 20261003T121202Z': 'Python 3.9.6, NumPy 1.24.4, SymPy 1.14.0, Qiskit 1.2.4, SciPy 1.10.1, macOS arm64',
}
MANIFEST_NAME = 'BUNDLE_MANIFEST.json'
EXCLUDE_TOP = {'runs'}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(value, fh, indent=2, ensure_ascii=False, sort_keys=False)
        fh.write('\n')


# ------------------------------------------------------------------------------------------
# env
# ------------------------------------------------------------------------------------------
def _module_version(name):
    try:
        mod = __import__(name)
        return getattr(mod, '__version__', 'unknown')
    except Exception as exc:  # pragma: no cover - depends on the host
        return 'MISSING (%s: %s)' % (type(exc).__name__, exc)


def cmd_env(a):
    versions = {'python': platform.python_version()}
    for name in ('numpy', 'sympy', 'qiskit', 'mpmath', 'scipy', 'rustworkx', 'symengine'):
        versions[name] = _module_version(name)
    warnings = []
    for key, want in EXPECTED.items():
        if versions.get(key) != want:
            warnings.append('%s %s != expected %s' % (key, versions.get(key), want))
    for key, want in INFORMATIONAL.items():
        if versions.get(key) != want:
            warnings.append('(informational) %s %s != %s' % (key, versions.get(key), want))
    missing = [k for k in ('numpy', 'sympy', 'qiskit', 'mpmath') if str(versions[k]).startswith('MISSING')]
    shim = sys.version_info < (3, 9) or os.environ.get('REPRO_FORCE_GCD_SHIM') == '1'
    rec = dict(
        checked_utc=utc_now(), executable=sys.executable, python_full=sys.version, platform=platform.platform(),
        machine=platform.machine(), cpu_count=os.cpu_count(), versions=versions, expected_server=EXPECTED,
        informational_pins=INFORMATIONAL, warnings=warnings, missing_required=missing,
        gcd_shim_will_be_used_for_kyriienko=shim, reference_run_environments=REFERENCE_RUNS,
        thread_env={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                                                    'RAYON_NUM_THREADS', 'PYTHONHASHSEED')})
    print('[env] python %s | numpy %s | sympy %s | qiskit %s | mpmath %s | scipy %s | rustworkx %s' % (
        versions['python'], versions['numpy'], versions['sympy'], versions['qiskit'], versions['mpmath'],
        versions['scipy'], versions['rustworkx']))
    print('[env] %s, %s CPUs, executable %s' % (rec['platform'], rec['cpu_count'], sys.executable))
    for w in warnings:
        print('[env] WARNING: version mismatch: %s (continuing; the comparison decides)' % w)
    for m in missing:
        print('[env] WARNING: required module %s is missing; the dependent stage will fail' % m)
    if shim:
        print('[env] note: Python < 3.9 -> the variadic math.gcd shim is used for kyriienko_scaling.py')
    if a.out:
        write_json(a.out, rec)
    return 0


# ------------------------------------------------------------------------------------------
# integrity / manifest
# ------------------------------------------------------------------------------------------
def _bundle_files(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        parts = [] if rel_dir == '.' else rel_dir.split(os.sep)
        if parts and parts[0] in EXCLUDE_TOP:
            dirnames[:] = []
            continue
        dirnames[:] = sorted(d for d in dirnames if d != '__pycache__' and not (not parts and d in EXCLUDE_TOP))
        for fn in sorted(filenames):
            if fn in ('.DS_Store',) or fn.endswith('.pyc'):
                continue
            rel = '/'.join(parts + [fn])
            if rel == MANIFEST_NAME:
                continue
            out.append(rel)
    return sorted(out)


def _original_for(rel):
    """Map a bundle path to its source path relative to journal_extension/ (or None)."""
    if rel.startswith('journal_extension/'):
        return rel[len('journal_extension/'):]
    pre = 'reference_results/fourier_gram/20261002T044925Z/'
    if rel.startswith(pre):
        return 'fourier_gram_identifiability_v1/experiment/results/20261002T044925Z/' + rel[len(pre):]
    pre = 'reference_results/kyriienko_scaling/20261003T121202Z/'
    if rel.startswith(pre):
        return 'kyriienko_scaling_v1/results/20261003T121202Z/' + rel[len(pre):]
    return None


def cmd_build_manifest(a):
    root = os.path.abspath(a.bundle)
    files = []
    for rel in _bundle_files(root):
        rec = dict(path=rel, sha256=sha256_file(os.path.join(root, rel)), bytes=os.path.getsize(os.path.join(root, rel)))
        orig = _original_for(rel)
        if orig is not None:
            rec['original'] = 'journal_extension/' + orig
            if a.source_root:
                src = os.path.join(a.source_root, orig)
                rec['original_sha256'] = sha256_file(src) if os.path.exists(src) else None
                rec['identical_to_original'] = rec['original_sha256'] == rec['sha256']
        else:
            rec['original'] = 'new in bundle'
        files.append(rec)
    copied = [f for f in files if 'identical_to_original' in f]
    man = dict(bundle='reproduction_bundle_v1', built_utc=utc_now(), file_count=len(files),
               copied_files=len(copied), copied_identical=all(f['identical_to_original'] for f in copied),
               note='journal_extension/ mirrors the needed subset of the original journal_extension tree so the '
                    'original scripts run unmodified; reference_results/ are the stored reference runs.',
               files=files)
    write_json(os.path.join(root, MANIFEST_NAME), man)
    print('[manifest] %d files, %d copied, all identical to originals: %s' % (
        len(files), len(copied), man['copied_identical']))
    return 0 if man['copied_identical'] else 1


def cmd_integrity(a):
    root = os.path.abspath(a.bundle)
    path = os.path.join(root, MANIFEST_NAME)
    if not os.path.exists(path):
        rep = dict(passed=False, error='BUNDLE_MANIFEST.json missing')
    else:
        with open(path, encoding='utf-8') as fh:
            man = json.load(fh)
        listed = {f['path']: f['sha256'] for f in man['files']}
        present = set(_bundle_files(root))
        missing = sorted(p for p in listed if p not in present)
        changed = sorted(p for p in listed if p in present and sha256_file(os.path.join(root, p)) != listed[p])
        extra = sorted(p for p in present if p not in listed)
        rep = dict(passed=not missing and not changed, files_listed=len(listed), missing=missing, changed=changed,
                   extra_unlisted=extra, manifest_sha256=sha256_file(path))
    rep['checked_utc'] = utc_now()
    print('[integrity] passed=%s missing=%d changed=%d extra=%d' % (
        rep['passed'], len(rep.get('missing', [])), len(rep.get('changed', [])), len(rep.get('extra_unlisted', []))))
    for p in rep.get('missing', []) + rep.get('changed', []):
        print('[integrity] PROBLEM: %s' % p)
    if a.out:
        write_json(a.out, rep)
    return 0


# ------------------------------------------------------------------------------------------
# run (stage runner)
# ------------------------------------------------------------------------------------------
def _maxrss_mb(ru):
    return ru.ru_maxrss / 2 ** 20 if sys.platform == 'darwin' else ru.ru_maxrss / 2 ** 10


def cmd_run(a):
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == '--' else a.cmd
    if not cmd:
        print('run: no command given', file=sys.stderr)
        return 2
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    start, t0 = utc_now(), time.perf_counter()
    print('[stage %s] start %s  cwd=%s' % (a.name, start, a.cwd or os.getcwd()), flush=True)
    with open(a.log, 'ab') as log:
        log.write(('# stage %s started %s\n# cmd: %s\n' % (a.name, start, ' '.join(cmd))).encode())
        try:
            proc = subprocess.Popen(cmd, cwd=a.cwd or None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        except OSError as exc:
            code = 127
            log.write(('could not start: %r\n' % (exc,)).encode())
            proc = None
        if proc is not None:
            for line in iter(proc.stdout.readline, b''):
                log.write(line)
                log.flush()
                sys.stdout.write(line.decode('utf-8', 'replace'))
                sys.stdout.flush()
            proc.stdout.close()
            code = proc.wait()
    wall = time.perf_counter() - t0
    peak = _maxrss_mb(resource.getrusage(resource.RUSAGE_CHILDREN))
    rec = dict(name=a.name, command=cmd, cwd=a.cwd or os.getcwd(), exit_code=code, started_utc=start,
               finished_utc=utc_now(), wall_seconds=round(wall, 3), peak_rss_mb_largest_process=round(peak, 1))
    status = {'stages': []}
    if os.path.exists(a.status):
        with open(a.status, encoding='utf-8') as fh:
            status = json.load(fh)
    status['stages'] = [s for s in status.get('stages', []) if s['name'] != a.name] + [rec]
    write_json(a.status, status)
    print('[stage %s] exit=%s wall=%.1fs peak_rss(largest process)=%.0f MB' % (a.name, code, wall, peak), flush=True)
    return code


# ------------------------------------------------------------------------------------------
# pyexec (Python 3.8 math.gcd shim)
# ------------------------------------------------------------------------------------------
def install_gcd_shim():
    import math
    if getattr(math.gcd, '_repro_variadic_shim', False):
        return
    gcd2 = math.gcd

    def gcd(*integers):
        if len(integers) == 2:
            return gcd2(integers[0], integers[1])
        g = 0
        for v in integers:
            g = gcd2(g, v)
        return g
    gcd._repro_variadic_shim = True
    math.gcd = gcd


def cmd_pyexec(rest):
    if rest and rest[0] == '--':
        rest = rest[1:]
    if not rest:
        print('pyexec: give a script or -m module', file=sys.stderr)
        return 2
    if sys.version_info < (3, 9) or os.environ.get('REPRO_FORCE_GCD_SHIM') == '1':
        install_gcd_shim()
        print('[pyexec] variadic math.gcd shim active (Python %s)' % platform.python_version(), file=sys.stderr, flush=True)
    if rest[0] == '-m':
        sys.path[0] = os.getcwd()
        sys.argv = [rest[1]] + rest[2:]
        runpy.run_module(rest[1], run_name='__main__', alter_sys=True)
    else:
        script = os.path.abspath(rest[0])
        sys.path[0] = os.path.dirname(script)
        sys.argv = [script] + rest[1:]
        runpy.run_path(script, run_name='__main__')
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'pyexec':
        return cmd_pyexec(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='subcommand')
    p = sub.add_parser('env')
    p.add_argument('--out')
    p = sub.add_parser('integrity')
    p.add_argument('--bundle', required=True)
    p.add_argument('--out')
    p = sub.add_parser('build-manifest')
    p.add_argument('--bundle', required=True)
    p.add_argument('--source-root')
    p = sub.add_parser('run')
    p.add_argument('--name', required=True)
    p.add_argument('--log', required=True)
    p.add_argument('--status', required=True)
    p.add_argument('--cwd')
    p.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    fn = {'env': cmd_env, 'integrity': cmd_integrity, 'build-manifest': cmd_build_manifest, 'run': cmd_run}.get(a.subcommand)
    if fn is None:
        ap.print_help()
        return 2
    return fn(a)


if __name__ == '__main__':
    sys.exit(main())
