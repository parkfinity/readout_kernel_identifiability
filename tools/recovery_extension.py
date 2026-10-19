"""Run the exact recovery checks and matched simulation without altering references."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE_ERROR_KEYS={'probability_factorization','ideal_features','probability_normalization',
                   'angle_derivative_finite_difference','tail_derivative_finite_difference'}

def check_source_errors(values):
    """Use the frozen engine's pre-existing 1e-8 validation threshold."""
    if set(values)!=SOURCE_ERROR_KEYS:
        raise ValueError('Source validation fields differ')
    for key,value in values.items():
        if type(value) is not float or not math.isfinite(value) or not 0<=value<=1e-8:
            raise ValueError('Source validation threshold exceeded: '+key)


def compare(left, right, path='root', rtol=1e-9, atol=1e-11):
    """Compare scientific JSON; integer ranks and all structural data stay exact."""
    if type(left) is not type(right):
        raise ValueError('JSON type mismatch: '+path)
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError('JSON keys mismatch: '+path)
        for key in left:
            compare(left[key],right[key],path+'/'+key,rtol,atol)
    elif isinstance(left,list):
        if len(left)!=len(right):
            raise ValueError('JSON length mismatch: '+path)
        for i,(a,b) in enumerate(zip(left,right)):
            compare(a,b,path+'/'+str(i),rtol,atol)
    elif isinstance(left,float):
        if not math.isfinite(left) or not math.isfinite(right) or not math.isclose(left,right,rel_tol=rtol,abs_tol=atol):
            raise ValueError('Scientific numeric mismatch: '+path)
    elif left != right:
        raise ValueError('Scientific value mismatch: '+path)


def run(root,out,replay=False):
    source=root/'experiments/minimal_product'
    work=out/'supplement'
    shutil.copytree(source,work)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',READOUT_KERNEL_PACKAGE=str(root),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    steps=[]
    with (out/'execution.log').open('w') as log:
        def command(arguments,cwd=work):
            print('[RUN]', ' '.join(arguments),flush=True)
            result=subprocess.run([sys.executable]+arguments,cwd=cwd,env=env,
                                  stdout=log,stderr=subprocess.STDOUT)
            log.flush()
            if result.returncode:
                raise RuntimeError('Check failed; see runs execution.log')
            steps.append(arguments)
        command(['-m','unittest','discover','-s','tools','-p','test_recovery_extension.py','-v'],root)
        command(['verify_supplement.py'])
        command(['-m','unittest','discover','-s','census_mapping','-v'])
        command(['census_mapping/verify_census.py','--package',str(root)])
        differences={}
        if replay:
            import numpy as np
            fresh=out/'fresh_experiment'
            fresh.mkdir()
            frozen=json.loads((source/'experiment/FROZEN_MANIFEST.json').read_text())
            for name in list(frozen)+['FROZEN_MANIFEST.json']:
                shutil.copy2(source/'experiment'/name,fresh/name)
            command(['run.py','run'],fresh)
            reference=source/'experiment/results'
            for name in ['records.json','source_checks.json','summary.json']:
                old=json.loads((reference/name).read_text())
                new=json.loads((fresh/'results'/name).read_text())
                if name=='source_checks.json':
                    for records in (old,new):
                        for record in records:
                            errors={key:record.pop(key) for key in SOURCE_ERROR_KEYS}
                            check_source_errors(errors)
                if name=='summary.json':
                    # Version, platform and executable are recorded separately, not science.
                    for data in (old,new):
                        data.pop('environment',None)
                        data.pop('elapsed_seconds',None)
                        check_source_errors(data.pop('source_check_max_errors'))
                compare(old,new,path=name)
            for name in ['exact_half_feature_covariances.npz','paired_distance_samples.npz']:
                maximum=0.0
                with np.load(reference/name,allow_pickle=False) as old, np.load(fresh/'results'/name,allow_pickle=False) as new:
                    if set(old.files)!=set(new.files):
                        raise ValueError('Array names differ: '+name)
                    for key in old.files:
                        np.testing.assert_allclose(old[key],new[key],rtol=1e-9,atol=1e-11,err_msg=name+':'+key)
                        maximum=max(maximum,float(np.max(np.abs(old[key]-new[key]))))
                differences[name]=maximum
            if (reference/'minimal_product_results.tex').read_bytes() != (fresh/'results/minimal_product_results.tex').read_bytes():
                raise ValueError('Generated rounded table differs')
    report=dict(passed=True,mode='minimal-product' if replay else 'minimal-product-check',
                exact_tests=17,experiment_tests=8,index_tests=6,integration_tests=8,census_settings=54,
                fresh_calculation=replay,records_replayed=480 if replay else 0,
                maximum_absolute_array_differences=differences,
                comparison_tolerance=dict(rtol=1e-9,atol=1e-11),
                metadata_excluded=['environment','elapsed_seconds'],
                source_validation_error_rule='Each error must meet frozen engine threshold 1e-8; roundoff error magnitudes need not match.',
                scope='Exact recovery algebra and ideal-model simulation; no QPU or full census generation.')
    (out/'verdict.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
