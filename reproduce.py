#!/usr/bin/env python3
"""Clean public entry point; reconstruct the hash-certified layout only inside runs/."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def safe_path(base, name):
    p = PurePosixPath(name)
    if not p.parts or p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Unsafe manifest path: '+name)
    target = base.joinpath(*p.parts)
    if base.resolve() not in (target.resolve(), *target.resolve().parents):
        raise ValueError('Path escapes root: '+name)
    return target

def save(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')

def integrity():
    manifest = json.loads((ROOT/'MANIFEST.json').read_text())['files']
    for name, expected in manifest.items():
        p = safe_path(ROOT,name)
        if not p.is_file() or digest(p) != expected:
            raise ValueError('File hash mismatch: '+name)
    mapping = json.loads((ROOT/'provenance/PATH_MAP.json').read_text())['files']
    original = set()
    for item in mapping:
        if item['original'] in original: raise ValueError('Duplicate original path')
        original.add(item['original'])
        safe_path(ROOT,item['original'])
        p = safe_path(ROOT,item['public'])
        if not p.is_file() or digest(p) != item['sha256']:
            raise ValueError('Source provenance mismatch: '+item['public'])
    return mapping,len(manifest)

def weight_three(out):
    """Fresh numerical calculation in public paths, with saved-array comparison."""
    import numpy as np
    experiment = ROOT/'experiments/kyriienko_weight3'
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    commands = [
        [sys.executable, '-m', 'unittest', 'discover', '-s', str(experiment),
         '-p', 'test_weight3.py', '-v'],
        [sys.executable, str(experiment/'run.py'), '--output', str(out/'results')]]
    with (out/'execution.log').open('w') as log:
        for command in commands:
            print('[RUN]', ' '.join(command), flush=True)
            code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            log.flush()
            if code:
                save(out/'verdict.json', dict(passed=False, mode='kyriienko-weight3', exit_code=code))
                raise RuntimeError('Numerical follow-up failed; see '+str(out/'execution.log'))
    ref = experiment/'reference_results'
    current = json.loads((out/'results/results.json').read_text())
    original = json.loads((ref/'results.json').read_text())
    assert current['passed'] and current['protocol'] == original['protocol']
    assert current['sources'] == original['sources']
    assert [s['ranks_by_relative_tolerance'] for s in current['settings']] == [
        s['ranks_by_relative_tolerance'] for s in original['settings']]
    assert digest(ref/'arrays.npz') == original['arrays_sha256']
    with np.load(out/'results/arrays.npz', allow_pickle=False) as a, np.load(ref/'arrays.npz', allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-9, atol=1e-9, err_msg=key)
    save(out/'verdict.json', dict(passed=True, mode='kyriienko-weight3',
        unit_tests=7, seeds=5, fresh_calculation=True, saved_array_comparison=True,
        result_directory='results', scope=current['scope']))
    print('PASS: '+str(out), flush=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['integrity','smoke','core-audit','core-full',
        'recovery','shots-pilot','shots-full','product','ad-summary','scaling-verify','kyriienko-weight3',
        'minimal-product-check','minimal-product'])
    ap.add_argument('--workers',type=int,default=2)
    args = ap.parse_args()
    if args.workers < 1: ap.error('--workers must be positive')
    if not __debug__ or os.environ.get('PYTHONOPTIMIZE'):
        raise RuntimeError('Do not disable scientific assertions with -O/PYTHONOPTIMIZE')
    mapping,n = integrity()
    print('Integrity OK:',n,'files;',len(mapping),'mapped source files verified',flush=True)
    if args.mode == 'integrity': return
    out = ROOT/'runs'/(args.mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True)
    if args.mode in ('minimal-product-check','minimal-product'):
        from tools.recovery_extension import run
        try:
            run(ROOT,out,replay=(args.mode == 'minimal-product'))
        except Exception as error:
            save(out/'verdict.json',dict(passed=False,mode=args.mode,error=str(error)))
            raise
        return
    if args.mode == 'kyriienko-weight3':
        weight_three(out)
        return
    work = out/'runtime'
    work.mkdir()
    for item in mapping:
        target = safe_path(work,item['original'])
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(safe_path(ROOT,item['public']),target)
        if digest(target) != item['sha256']: raise ValueError('Runtime copy mismatch')
    if args.mode == 'scaling-verify':
        core = work/'core_reproduction/core'
        command = [sys.executable,
            str(core/'journal_extension/kyriienko_scaling_v1/kyriienko_scaling.py'),
            '--verify',str(core/'reference_results/kyriienko_scaling/20261003T121202Z')]
    else:
        command = [sys.executable,str(work/'reproduce.py'),args.mode,'--workers',str(args.workers)]
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    print('[RUN]',args.mode,'(isolated original-source layout)',flush=True)
    with (out/'execution.log').open('w') as log:
        proc = subprocess.Popen(command,cwd=work,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
        try:
            for line in proc.stdout:
                log.write(line); log.flush(); print(line,end='',flush=True)
            code = proc.wait()
        except BaseException:
            os.killpg(proc.pid,signal.SIGTERM); proc.wait()
            save(out/'verdict.json',dict(passed=False,mode=args.mode,interrupted=True))
            raise
    if args.mode == 'scaling-verify':
        try:
            result = json.loads((out/'execution.log').read_text())
            verified = (code == 0 and result.get('passed') is True
                        and len(result.get('settings',[])) == 12)
        except (ValueError, OSError):
            verified = False
        save(out/'verdict.json',dict(passed=verified,mode=args.mode,exit_code=code,
            scope='Replay of 12 stored determinant certificates, not fresh circuit generation.'))
        print(('PASS: ' if verified else 'FAIL: ')+str(out),flush=True)
        if not verified: sys.exit(code or 1)
        return
    records = sorted((work/'runs').glob(args.mode+'_*/verdict.json'))
    verified = code == 0 and len(records) == 1 and json.loads(records[0].read_text()).get('passed') is True
    if records:
        for name in ('environment.json','stages.json'):
            src = records[-1].parent/name
            if src.is_file(): shutil.copy2(src,out/name)
    save(out/'verdict.json',dict(passed=verified,mode=args.mode,exit_code=code,
        source_layout_reconstruction=True,source_files_match_release_manifest=True,
        compatibility_patch_record='provenance/COMPATIBILITY_CHANGES.json',
        result_directory=str(records[0].parent.relative_to(out)) if len(records)==1 else None,
        scope='Only the selected workflow was run; no new science or altered reference evidence.'))
    print(('PASS: ' if verified else 'FAIL: ')+str(out),flush=True)
    if not verified: sys.exit(code or 1)

if __name__ == '__main__': main()
