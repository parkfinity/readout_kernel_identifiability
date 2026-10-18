"""Post-pilot: recover local-Z feature directions by retaining joint parities."""
import argparse
import json
from fractions import Fraction as F
from pathlib import Path
from design import HERE,fg,words,measured
from exact_controls import certify
from experiment import save

def run():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',type=Path);args=ap.parse_args()
    root=args.pilot or sorted((HERE/'results').glob('*/readout_summary.json'))[-1].parent
    path=HERE/'frozen_inputs/hubregtsen_encoding_circuit_n4_l1_local_z.json'
    info=json.loads(path.read_text());bank=words(4);idx=measured(bank,['ZZZZ'])
    info['observables']=[bank[i] for i in idx];info['weights']=[1]*len(idx)
    bases,angles=fg.ex.canonical_angles(info)
    # Exact formal expansions prove that the last four source symbols are
    # absent from ALL measured features, so any resulting kernel has rank <=4.
    for obs in info['observables']:
        terms=fg.ex.feature(info,obs,angles)
        assert all(t<4 for phase in terms for t,x,c in phase)
    raw=[F(len(bank)-1+3*i,len(bank)-1) for i in idx];weights=[w/sum(raw) for w in raw]
    row=certify(info,'hubregtsen_n4_l1_Z_and_ZZ_correlations',weights,['ZZZZ'])
    assert row['lower_bound']==4
    row['upper_bound']=4;row['certified_generic_rank']=4
    row['upper_proof']='Exact feature expansions contain no theta_4,...,theta_7; also all final CRZ gates commute with Z strings.'
    row['comparison']=dict(local_Z_single_features=4,augmented_Z_features=10,
                          initial_basis_settings=1,augmented_basis_settings=1,
                          local_Z_diagonal_metric_rank=2,local_Z_best_dense_metric_rank=3,
                          augmented_readout_generic_rank=4)
    row['scope']='Additional correlations require joint bit strings or joint estimates; not reconstructible from saved marginal means alone. Rank recovery, not equal-precision shot reduction.'
    save(root/'correlation_control.json',row)
    print(json.dumps({k:v for k,v in row.items() if k not in ('witnesses','parameter_point')},indent=2))

if __name__=='__main__':run()
