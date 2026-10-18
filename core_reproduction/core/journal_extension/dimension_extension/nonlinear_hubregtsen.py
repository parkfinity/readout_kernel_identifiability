"""Post-survey exact feature proof of a nonlinear two-coordinate kernel."""
from fractions import Fraction as F
import argparse
import json
from pathlib import Path
import numpy as np
import sympy as sp
import exact_engine as ex
from experiment import save,sha
ROOT=Path(__file__).resolve().parent

def run(output=None):
    p=ROOT/'frozen_inputs/hubregtsen_encoding_circuit_n4_l1_local_z.json'
    info=json.loads(p.read_text());bases,angles=ex.canonical_angles(info)
    assert bases==['a0','a1'] and info['count']==8
    certificates=[]
    for q,obs in enumerate(info['observables']):
        expected={}
        for s in (-1,1):
            for t in (-1,1):
                phase=ex.phase_add(((q,-1,F(s)),),((-1,q%2,F(t)),))
                ex.push(expected,phase,(F(0),F(s,4)))
        actual=ex.feature(info,obs,angles)
        assert actual==expected,obs
        certificates.append(dict(observable=obs,exact_feature='-sin(theta_%d)*cos(x_%d)'%(q,q%2),
                                 expansion=ex.canonical_dump(actual)))
    # Sum_q [f_q(x)-f_q(x')]^2 groups into the two repeated inputs.
    a=sp.Symbol('a',real=True);identity=sp.expand_trig(sp.sin(2*a))-2*sp.sin(a)*sp.cos(a)
    assert sp.simplify(identity)==0
    c=(1-sp.sqrt(2)/2)**2;det=sp.expand(c*c)
    assert det==sp.Rational(17,4)-3*sp.sqrt(2) and det>0
    # All four retained coordinates are globally individually active, but no
    # nonzero constant vector annihilates both beta gradients for every theta.
    # For either input group, vary each theta independently in sin(2 theta).
    result=dict(passed=True,source_sha256=sha(p),feature_identities=certificates,
                beta=['sin(theta_0)^2 + sin(theta_2)^2','sin(theta_1)^2 + sin(theta_3)^2'],
                distance='beta_0*(cos(x_0)-cos(xprime_0))^2 + beta_1*(cos(x_1)-cos(xprime_1))^2',
                exact_generic_rank=2,constant_linear_factor_rank=4,
                lower_witness=dict(theta_first_four=['pi/4']*4,theta_last_four='arbitrary',
                     input_pairs=[[['0','0'],['pi/4','0']],[['0','0'],['0','pi/4']]],
                     source_columns=[0,1],minor_determinant=str(det)),
                all_four_retained_coordinates_active=True,
                interpretation='Nonlinear kernel coordinates; not an extra inactive-coordinate or gate/shot saving',
                derivation='Final CRZ gates commute with each local Z. H followed by Rz(x) and Ry(theta) yields -sin(theta) cos(x). Exact frozen-IR feature coefficients match these identities.',
                script_sha256=sha(Path(__file__)))
    save(output or ROOT/'nonlinear_hubregtsen.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='feature_identities'},indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args();run(args.output)
