"""Exact sufficient identities for a deliberately restricted rational-angle IR.

No independence theorem for exponentials of polynomial phases is assumed:
zero formal coefficients suffice; nonzero ones do NOT certify activity.
Qubit words use ascending wire order. Gaussian-rational coefficients are pairs.
"""
from collections import defaultdict
from fractions import Fraction as F
import hashlib
import json
import sympy as sp

ZERO = (F(0), F(0))
ONE = (F(1), F(0))


def add(a, b): return a[0]+b[0], a[1]+b[1]
def mul(a, b): return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]
def scale(a, b): return a[0]*b, a[1]*b


def push(d, key, value):
    z = add(d.get(key, ZERO), value)
    if z == ZERO: d.pop(key, None)
    else: d[key] = z


def phase_add(a, b, sign=1):
    d = defaultdict(F)
    for t, x, c in a: d[t, x] += c
    for t, x, c in b: d[t, x] += sign*c
    return tuple((t, x, c) for (t, x), c in sorted(d.items()) if c)


def product(a, b):
    d = {'XY': ((0, 1), 'Z'), 'YX': ((0, -1), 'Z'),
         'YZ': ((0, 1), 'X'), 'ZY': ((0, -1), 'X'),
         'ZX': ((0, 1), 'Y'), 'XZ': ((0, -1), 'Y')}
    out = []; phase = ONE
    for x, y in zip(a, b):
        if x == 'I': out.append(y)
        elif y == 'I': out.append(x)
        elif x == y: out.append('I')
        else:
            z, w = d[x+y]; phase = mul(phase, z); out.append(w)
    return phase, ''.join(out)


def commutes(a, b):
    return sum(x != 'I' and y != 'I' and x != y for x, y in zip(a, b)) % 2 == 0


def conjugate(word, gate):
    if gate['op'] == 'H':
        q = gate['wires'][0]; w = list(word); a = w[q]
        w[q] = dict(I='I', X='Z', Y='Y', Z='X')[a]
        return (-1 if a == 'Y' else 1), ''.join(w)
    if gate['op'] != 'CX': raise ValueError('Unsupported Clifford')
    c, t = gate['wires']; out = 'I'*len(word); phase = ONE
    for q, a in enumerate(word):
        if a == 'I': continue
        if a == 'Y': phase = mul(phase, (0, 1))
        for axis in ('XZ' if a == 'Y' else a):
            w = ['I']*len(word); w[q] = axis
            if axis == 'X' and q == c: w[t] = 'X'
            if axis == 'Z' and q == t: w[c] = 'Z'
            z, out = product(out, ''.join(w)); phase = mul(phase, z)
    assert phase[1] == 0 and abs(phase[0]) == 1
    return phase[0], out


def validate(info):
    if info['n'] not in (1, 2, 4): raise ValueError('Unsupported pilot width')
    if len(set(info['observables'])) != len(info['observables']): raise ValueError('Repeated readout')
    if any(F(str(w)) <= 0 for w in info['weights']): raise ValueError('Invalid metric')
    if len(info['weights']) != len(info['observables']): raise ValueError('Metric mismatch')
    # No approximate restoration or undeclared arbitrary-constant semantics.
    for g in info['program']:
        if g['op'] == 'R':
            if len(g['pauli']) != info['n'] or set(g['pauli'])-set('IXYZ'): raise ValueError('Bad Pauli')
            for c, t, x in g['terms']:
                if not float(c).is_integer(): raise ValueError('Pilot accepts integer IR coefficients only')
                if not -1 <= t < info['count'] or not -1 <= x < len(info['data_expressions']):
                    raise ValueError('Bad source index')
        elif g['op'] not in ('H', 'CX'): raise ValueError('Unsupported gate')
        else:
            if len(g['wires']) != (1 if g['op'] == 'H' else 2): raise ValueError('Bad wires')
            if len(set(g['wires'])) != len(g['wires']) or any(not 0 <= q < info['n'] for q in g['wires']):
                raise ValueError('Bad wires')
    for expression in info['data_expressions']:
        c, base = sp.sympify(expression, rational=True).as_coeff_Mul()
        if not c.is_Rational or str(base) not in ('a0', 'a1', 'acos(a0)', 'acos(a1)'):
            raise ValueError('Unsupported data expression: '+expression)


def canonical_angles(info):
    validate(info)
    bases = []; slots = []
    for expression in info['data_expressions']:
        c, base = sp.sympify(expression, rational=True).as_coeff_Mul()
        name = str(base)
        if name not in bases: bases.append(name)
        slots.append((F(str(c)), bases.index(name)))
    result = []
    for g in info['program']:
        if g['op'] != 'R': result.append(None); continue
        d = defaultdict(F)
        for c, t, x in g['terms']:
            factor, slot = slots[x] if x >= 0 else (F(1), -1)
            d[t, slot] += F(int(c))*factor
        result.append(tuple((t, x, c) for (t, x), c in sorted(d.items()) if c))
    return bases, result


def feature(info, observable, angles):
    work = {(observable, ()): ONE}
    for g, angle in reversed(list(zip(info['program'], angles))):
        new = {}
        for (word, phase), coeff in work.items():
            if g['op'] != 'R':
                sign, w = conjugate(word, g); push(new, (w, phase), scale(coeff, sign))
            elif commutes(word, g['pauli']): push(new, (word, phase), coeff)
            else:
                z, w = product(g['pauli'], word)
                # U† O U = cos(alpha) O + i sin(alpha) P O.
                for direction in (-1, 1):
                    ph = phase_add(phase, angle, direction)
                    push(new, (word, ph), scale(coeff, F(1, 2)))
                    push(new, (w, ph), scale(mul(coeff, z), F(direction, 2)))
        work = new
        if len(work) > 250000: raise RuntimeError('Symbolic term budget exceeded')
    result = {}
    for (word, phase), coeff in work.items():
        if set(word) <= set('IZ'): push(result, phase, coeff)
    return result


def distance(info):
    bases, angles = canonical_angles(info); result = {}
    for obs, weight in zip(info['observables'], info['weights']):
        f = feature(info, obs, angles); delta = dict(f)
        for ph, c in f.items():
            prime = tuple((t, x+len(bases) if x >= 0 else -1, v) for t, x, v in ph)
            push(delta, prime, scale(c, -1))
        for a, ca in delta.items():
            for b, cb in delta.items():
                push(result, phase_add(a, b), scale(mul(ca, cb), F(str(weight))))
        if len(result) > 250000: raise RuntimeError('Distance term budget exceeded')
    return bases, result


def derivatives(d, count):
    result = [{} for _ in range(count)]
    for phase, coefficient in d.items():
        for t, x, c in phase:
            if t >= 0: push(result[t], (phase, x), scale(mul(coefficient, (0, 1)), c))
    return result


def formal_rank(deriv):
    count = len(deriv); rows = {}
    for k, column in enumerate(deriv):
        for key, c in column.items():
            for part in (0, 1):
                if c[part]: rows.setdefault((key, part), [F(0)]*count)[k] = c[part]
    basis = {}
    for row in rows.values():
        r = row[:]
        for p, b in sorted(basis.items()):
            if r[p]:
                factor = r[p]; r = [a-factor*v for a, v in zip(r, b)]
        nz = next((i for i, v in enumerate(r) if v), None)
        if nz is not None:
            value = r[nz]; basis[nz] = [v/value for v in r]
    m = sp.Matrix(list(basis.values())) if basis else sp.zeros(0, count)
    nulls = [[str(v) for v in vec] for vec in m.nullspace()]
    return len(basis), nulls


def canonical_dump(d):
    return [[[[t, x, str(c)] for t, x, c in ph], [str(c[0]), str(c[1])]]
            for ph, c in sorted(d.items())]


def certify(info):
    bases, d = distance(info); deriv = derivatives(d, info['count'])
    upper, nulls = formal_rank(deriv)
    for v in nulls:
        check = {}
        for c, col in zip(v, deriv):
            for key, z in col.items(): push(check, key, scale(z, F(c)))
        assert not check
    payload = canonical_dump(d)
    return {'r_upper': upper, 'null_basis': nulls,
            'inactive': [k for k, col in enumerate(deriv) if not col],
            'distance_terms': len(d), 'basis_data': bases,
            'distance_sha256': hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
            'distance_is_zero': not d}, d
