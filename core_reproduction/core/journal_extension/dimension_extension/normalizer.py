"""Signed Clifford extraction and commuting Pauli fusion, not QuCLEAR/Paulihedral.

No approximate equality, synthesis heuristic, or numerical rank threshold is used.
The trailing Clifford is retained unless its exact tableau is identity.
"""
import copy
from collections import defaultdict
from fractions import Fraction
import numpy as np
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Pauli
from structural_core import commute
from certificates import invariant_generator


def frame_circuit(frame, n):
    q = QuantumCircuit(n)
    for g in frame:
        if g['op'] == 'H': q.h(g['wires'][0])
        elif g['op'] == 'CX': q.cx(*g['wires'])
        else: raise ValueError(g)
    return q


def signed_word(word, frame, n, direction):
    p = Pauli(word[::-1]).evolve(Clifford(frame_circuit(frame,n)), frame=direction)
    phase = int(p.phase)
    if phase not in (0,2): raise ValueError('Non-Hermitian generator')
    p.phase = 0
    return p.to_label()[::-1], (1 if phase == 0 else -1)


def collect(terms):
    d = defaultdict(Fraction)
    for c,t,x in terms: d[t,x] += Fraction(str(c))
    if any(Fraction(str(float(c)))!=c for c in d.values()):
        raise ValueError('Merged coefficient cannot be stored exactly as a decimal float')
    return [[float(c),t,x] for (t,x),c in sorted(d.items()) if c]


def normalize(info):
    frame=[]; rotations=[]; log=[]; n=info['n']
    for index,g in enumerate(info['program']):
        if g['op'] in ('H','CX'):
            frame.append(copy.deepcopy(g)); continue
        if g['op'] != 'R': raise ValueError(g)
        # U_prefix = F R. Next Q gives Q F R = F (F† Q F) R.
        word,sign=signed_word(g['pauli'],frame,n,'h')
        v={'op':'R','pauli':word,'terms':[[sign*c,t,x] for c,t,x in g['terms']]}
        target=None
        for j in range(len(rotations)-1,-1,-1):
            h=rotations[j]
            if h['pauli']==word: target=j; break
            if not commute(h['pauli'],word): break
        log.append({'input_index':index,'frame':copy.deepcopy(frame),
                    'input_word':g['pauli'],'output_word':word,'sign':sign,'merge_target':target})
        if target is None: rotations.append(v)
        else: rotations[target]['terms'].extend(v['terms'])
    for g in rotations: g['terms']=collect(g['terms'])
    rotations=[g for g in rotations if g['terms']]
    identity=Clifford(frame_circuit(frame,n))==Clifford(QuantumCircuit(n))
    tail=[] if identity else frame
    out=dict(info,program=rotations+tail)
    out['ids']=sorted({t for g in rotations for c,t,x in g['terms'] if t>=0})
    return out,{'rewrites':log,'trailing_frame':tail,'frame_identity_up_to_phase':bool(identity)}


def delete_common_isometries(info):
    out=copy.deepcopy(info); p=out['program']; removed=[]
    for i,g in enumerate(p):
        if g['op']!='R': continue
        word=g['pauli']; safe=True
        for later in p[i+1:]:
            if later['op']=='R':
                if not commute(word,later['pauli']): safe=False; break
            else: word,_=signed_word(word,[later],info['n'],'s')
        if safe and invariant_generator(word,info['observables'],info['weights'])[0]:
            discarded=[t for t in g['terms'] if t[2]==-1]
            if discarded: removed.append({'rotation_index':i,'output_word':word,'terms':discarded})
            g['terms']=[t for t in g['terms'] if t[2]!=-1]
    out['program']=[g for g in p if g['op']!='R' or g['terms']]
    out['ids']=sorted({t for g in out['program'] if g['op']=='R' for c,t,x in g['terms'] if t>=0})
    return out,removed


def coefficient_rank(info):
    """Exact linear parameter-combination rank of angles, NOT kernel rank."""
    expr=info['data_expressions']; rows=[]
    for g in info['program']:
        if g['op']!='R': continue
        table={}
        for c,t,x in g['terms']:
            if t<0: continue
            key='1' if x<0 else str(sp.simplify(sp.sympify(expr[x])))
            row=table.setdefault(key,[sp.S.Zero]*info['count'])
            row[t]+=sp.Rational(str(c))
        rows.extend(table.values())
    return int(sp.Matrix(rows).rank()) if rows else 0


def resources(info):
    rotations=[g for g in info['program'] if g['op']=='R']
    angles=sum(any(t>=0 for c,t,x in g['terms']) for g in rotations)
    return {'symbols':len(info['ids']),'pauli_rotation_tasks':len(rotations),
            'trainable_angle_tasks':angles,'trailing_clifford_gates':len(info['program'])-len(rotations),
            'angle_coefficient_rank':coefficient_rank(info)}
