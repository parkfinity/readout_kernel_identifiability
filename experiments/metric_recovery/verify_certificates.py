"""Recheck exact minors, including independent SymPy rational determinants."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np
import sympy as sp
from design import HERE,fg
from experiment import save

def run():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',type=Path);args=ap.parse_args()
    root=args.pilot or sorted((HERE/'results').glob('*/readout_summary.json'))[-1].parent
    summary=json.loads((root/'exact_controls/summary.json').read_text())
    checked=0
    for row in summary['controls']:
        for w in row['witnesses']:
            value=int(sp.Matrix(w['minor_matrix']).det(method='domain-ge'))%w['prime']
            assert value==w['determinant_mod_prime'] and value!=0
            assert len(w['minor_columns'])==row['K']
            checked+=1
    rational=[]
    for prefix in ('hubregtsen_encoding_circuit','yz_cx_encoding_circuit','kyriienko_nonlinear_encoding_circuit'):
        name=prefix+'_n2_l1_local_xyz'
        row=next(r for r in summary['controls'] if r['setting']==name+'_fixed_diagonal')
        st=fg.Setting(json.loads((HERE/'frozen_inputs'/(name+'.json')).read_text()))
        eng=fg.LaurentEngine(st,row['parameter_point'],mode='exact').run()
        C,dC,labels=eng.coefficients_exact();li={label:i for i,label in enumerate(labels)}
        weights=list(map(F,row['metric_weights']));minor=[]
        w=row['witnesses'][0]
        for a,b in w['gram_coordinates']:
            ia,ib=li[st.basis[a]],li[st.basis[b]]
            minor.append([sum(weights[o]*(dC[k][o][ia]*C[o][ib]+C[o][ia]*dC[k][o][ib]) for o in range(len(C))) for k in w['minor_columns']])
        det=sp.Matrix(minor).det(method='domain-ge');assert det!=0
        for witness in row['witnesses']:
            # The saved pivot rows are identical across these two primes.
            if witness['gram_coordinates']==w['gram_coordinates'] and witness['minor_columns']==w['minor_columns']:
                assert fg.frac_mod(F(int(sp.numer(det)),int(sp.denom(det))),witness['prime'])==witness['determinant_mod_prime']
        rational.append(dict(setting=name,exact_rank=st.K,exact_rational_minor=str(det)))
    result=dict(passed=True,independent_integer_minor_checks=checked,exact_rational_witnesses=rational,
                scope='Exact arithmetic implementation checks, not a formal proof assistant or independent research review')
    save(root/'certificate_recheck.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':run()
