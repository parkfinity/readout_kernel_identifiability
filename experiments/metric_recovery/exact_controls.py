"""Exact lower witnesses and analytical upper bounds; no floating zero proofs."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import sympy as sp
from design import fg,words,measured,HERE

def save(path,obj): path.write_text(json.dumps(obj,indent=2,default=str)+'\n')

def hub_control():
    info=json.loads((HERE/'frozen_inputs/hubregtsen_encoding_circuit_n4_l1_local_z.json').read_text())
    ex=fg.ex; bases,angles=ex.canonical_angles(info)
    assert bases==['a0','a1'] and info['count']==8
    identities=[]
    for q,o in enumerate(info['observables']):
        expected={}
        for s in (-1,1):
            for t in (-1,1):
                phase=ex.phase_add(((q,-1,F(s)),),((-1,q%2,F(t)),))
                ex.push(expected,phase,(F(0),F(s,4)))
        assert ex.feature(info,o,angles)==expected
        identities.append('-sin(theta_%d)*cos(x_%d)'%(q,q%2))
    a,b,c,d=sp.symbols('a b c d',real=True)
    C=sp.Matrix([[a,0],[0,b],[c,0],[0,d]])
    u=sp.Matrix([1,2,3,4]); M=sp.eye(4)+u*u.T/32; M=M/sp.trace(M)
    G=C.T*M*C; J=sp.Matrix([G[0,0],G[0,1],G[1,1]]).jacobian([a,b,c,d])
    # a,b,c,d = sin(theta_i); both sine and cosine are exact rationals.
    subs={a:sp.Rational(3,5),b:sp.Rational(5,13),c:sp.Rational(8,17),d:sp.Rational(7,25)}
    cos=sp.diag(sp.Rational(4,5),sp.Rational(12,13),sp.Rational(15,17),sp.Rational(24,25))
    JT=J.subs(subs)*cos
    cols=list(JT.rref()[1]);minor=JT[:,cols].det()
    assert len(cols)==3 and minor!=0
    w=sp.symbols('w0:4',positive=True); GD=C.T*sp.diag(*w)*C
    assert GD[0,1]==0
    JD=sp.Matrix([GD[0,0],GD[1,1]]).jacobian([a,b,c,d])
    diagonal_minor=sp.factor(JD[:,[0,1]].det()); assert diagonal_minor==4*a*b*w[0]*w[1]
    return dict(passed=True,source_sha256=hashlib.sha256((HERE/'frozen_inputs/hubregtsen_encoding_circuit_n4_l1_local_z.json').read_bytes()).hexdigest(),
                feature_identities=identities,nonconstant_data_basis=['cos(x0)','cos(x1)'],
                any_fixed_diagonal_SPD_generic_rank=2,
                chosen_dense_metric=[[str(x) for x in row] for row in M.tolist()],
                chosen_dense_metric_generic_rank=3,
                any_fixed_dense_metric_rank_upper=3,
                centered_feature_generic_rank=4,
                upper_reason='A=2 nonconstant data basis; symmetric coefficient Gram has only 3 entries',
                lower_minor_columns=cols,lower_minor_determinant=str(minor),
                sine_values=[str(subs[x]) for x in (a,b,c,d)],cosine_values=[str(cos[i,i]) for i in range(4)],
                gram_entries=[str(G[i,j]) for i,j in [(0,0),(0,1),(1,1)]],
                diagonal_minor=str(diagonal_minor),
                implication='No single fixed metric on these four Z features can recover all four feature directions; extra information or a different model is necessary.')

def certify(info,name,weights,bases=None):
    st=fg.Setting(info,name); point=fg.pythagorean_point(st,2026100441)
    records=[]; t0=time.perf_counter()
    for p in fg.PRIMES:
        eng=fg.LaurentEngine(st,point,mode='mod',p=p).run()
        C,dC,real_ok=eng.coefficients_mod(); assert real_ok
        J,support,iu=fg.gram_jacobian_mod(C,dC,weights,p)
        rank,rows,cols=fg.rank_mod(J,p)
        minor=J[np.ix_(rows,cols)];det=fg.det_mod(minor.tolist(),p)
        assert rank==0 or det!=0
        records.append(dict(prime=p,rank_lower_bound=rank,minor_rows=rows,minor_columns=cols,
                            determinant_mod_prime=det,minor_matrix=minor.tolist(),
                            gram_coordinates=[[int(support[iu[0][r]]),int(support[iu[1][r]])] for r in rows]))
    lb=max(r['rank_lower_bound'] for r in records)
    return dict(setting=name,K=st.K,measurement_bases=bases,observables=st.obs,
                metric_weights=[str(w) for w in weights],parameter_point=point,
                lower_bound=lb,upper_bound=st.K,
                certified_generic_rank=lb if lb==st.K else None,
                arithmetic='Exact nonzero finite-field minors of rational Fourier coefficients. Finite-field zero is not an upper bound.',
                witnesses=records,elapsed_seconds=time.perf_counter()-t0)

def run():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',type=Path);ap.add_argument('--output',type=Path);args=ap.parse_args()
    pilot=args.pilot or sorted((HERE/'results').glob('*/readout_summary.json'))[-1].parent
    out=args.output or pilot/'exact_controls'
    out.mkdir(parents=True,exist_ok=False)
    hub=hub_control();save(out/'hubregtsen_metric_ceiling.json',hub)
    rows=[]
    # Added after the numerical pilot: verification of all ten designed schedules,
    # not selection of successful settings. Scope and timing are explicitly saved.
    for path in sorted((HERE/'frozen_inputs').glob('*local_xyz.json')):
        info=json.loads(path.read_text());O=len(info['observables'])
        raw=[F(O-1+3*i,O-1) for i in range(O)];weights=[w/sum(raw) for w in raw]
        row=certify(info,path.stem+'_fixed_diagonal',weights)
        rows.append(row);save(out/(row['setting']+'.json'),row)
        print(row['setting'],row['lower_bound'],row['upper_bound'],flush=True)
    for record in json.loads((pilot/'readout_summary.json').read_text()):
        path=HERE/'frozen_inputs'/(record['circuit']+'_local_xyz.json');info=json.loads(path.read_text())
        selection=next(r for r in record['records'] if r['method']=='rank_greedy' and r['budget']==1)
        bank=words(info['n']);idx=measured(bank,selection['bases'])
        raw=[F(len(bank)-1+3*i,len(bank)-1) for i in idx];weights=[w/sum(raw) for w in raw]
        info['observables']=[bank[i] for i in idx];info['weights']=[1]*len(idx)
        row=certify(info,record['circuit']+'_selected_one_basis',weights,selection['bases'])
        rows.append(row);save(out/(row['setting']+'.json'),row)
        print(row['setting'],row['lower_bound'],row['upper_bound'],flush=True)
    save(out/'summary.json',dict(controls=rows,hubregtsen=hub,
                exact_full_rank_count=sum(r['certified_generic_rank']==r['K'] for r in rows),
                total=len(rows),scope='Post-pilot exact verification of every circuit, not an independently preregistered new survey'))

if __name__=='__main__':run()
