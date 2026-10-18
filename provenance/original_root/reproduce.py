#!/usr/bin/env python3
"""Non-destructive entry point for the integrated October 2026 release."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')

def integrity():
    manifest = json.loads((ROOT/'MANIFEST.json').read_text())
    for name, digest in manifest['files'].items():
        p = ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('Integrity mismatch: ' + name)
    return len(manifest['files'])

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['integrity','smoke','core-audit','core-full',
        'recovery','shots-pilot','shots-full','product','ad-summary'])
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    if not __debug__ or os.environ.get('PYTHONOPTIMIZE'):
        raise RuntimeError('Run without -O/PYTHONOPTIMIZE: scientific assertions are required')
    count = integrity()
    print('Integrity OK:', count, 'files', flush=True)
    if args.mode == 'integrity':
        return
    out = ROOT/'runs'/(args.mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True)
    versions = {}
    for package in ('numpy','sympy','scipy','qiskit'):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    save(out/'environment.json', dict(python=sys.version, platform=platform.platform(), packages=versions))
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    stages = []
    def stage(name, cwd, *cmd):
        log = out/(name+'.log')
        print('[RUN]',name, flush=True)
        started = time.monotonic()
        with log.open('w') as f:
            proc = subprocess.Popen(list(cmd), cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
            try:
                while True:
                    try: code = proc.wait(timeout=15); break
                    except subprocess.TimeoutExpired: print('[WORKING]',name, flush=True)
            except BaseException:
                proc.terminate(); proc.wait(); raise
        stages.append(dict(stage=name, exit_code=code, seconds=time.monotonic()-started, log=log.name))
        save(out/'stages.json',stages)
        if code:
            print(log.read_text()[-5000:])
            raise RuntimeError(name+' failed; see '+str(log))
        print('[PASS]',name, flush=True)
    try:
        if args.mode in ('smoke','core-audit','core-full'):
            core_mode = {'smoke':'smoke','core-audit':'audit','core-full':'full'}[args.mode]
            stage('core_'+core_mode, ROOT/'core_reproduction', sys.executable,'review.py',core_mode,
                  '--workers',str(args.workers),'--out',str(out/'core_output'))
        if args.mode not in ('core-audit','core-full','ad-summary'):
            work = out/'work'
            shutil.copytree(ROOT/'experiments',work)
            v1 = work/'readout_metric_design_v1'
            v2 = work/'readout_metric_shot_design_v2'
            prod = work/'readout_product_baseline_v1'
            if args.mode == 'smoke':
                for name,cwd,test in [('recovery_tests',v1,'test_design'),
                                     ('shots_tests',v2,'test_shots'),('product_tests',prod,'test_product')]:
                    stage(name,cwd,sys.executable,'-m','unittest','-v',test)
                stage('frozen_plan',v2,sys.executable,'-c','from core import load_plan; load_plan()')
            elif args.mode == 'recovery':
                for script in ('experiment.py','exact_controls.py','verify_certificates.py',
                               'followup.py','correlation_control.py'):
                    stage(script[:-3],v1,sys.executable,script)
            elif args.mode in ('shots-pilot','shots-full'):
                stage('shots',v2,sys.executable,'experiment.py','--mode',args.mode.split('-')[1])
                stage('analysis',v2,sys.executable,'analyze.py')
            elif args.mode == 'product':
                stage('product',prod,sys.executable,'run.py')
        if args.mode == 'ad-summary':
            stage('ad_category_reaggregation',ROOT,sys.executable,'tools/check_ad_summary.py',str(out/'ad_check.json'))
        save(out/'verdict.json',dict(passed=True,mode=args.mode,
             scope='Only the selected workflow was executed; read README for exclusions.'))
        print('PASS:',out, flush=True)
    except BaseException as exc:
        save(out/'verdict.json',dict(passed=False,mode=args.mode,error=str(exc)))
        raise

if __name__ == '__main__':
    main()
