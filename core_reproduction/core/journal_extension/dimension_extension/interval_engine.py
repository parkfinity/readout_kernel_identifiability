"""Independent interval statevector/tangent witness checker for the pilot IR.

No Fourier engine imports. All input sharing is evaluated from original x, theta.
mpmath interval arithmetic is experimental software: this is a checked numerical
enclosure implementation, not a proof-assistant-verified arithmetic library.
"""
from fractions import Fraction as F
import hashlib
import json
from mpmath import iv
import sympy as sp


def rational(v):
    x = F(str(v)); return iv.mpf(x.numerator)/x.denominator


def pauli(state, word):
    out = [iv.mpc(0) for _ in state]
    for j, v in enumerate(state):
        target = j; phase = 1
        for q, a in enumerate(word):
            bit = (j >> q) & 1
            if a in 'XY': target ^= 1 << q
            if a == 'Y': phase *= -1j if bit else 1j
            if a == 'Z' and bit: phase *= -1
        out[target] += phase*v
    return out


def clifford(state, gate):
    if gate['op'] == 'H':
        bit = 1 << gate['wires'][0]; root = iv.sqrt(2)
        return [(state[j ^ bit]-state[j] if j & bit else state[j]+state[j ^ bit])/root
                for j in range(len(state))]
    if gate['op'] != 'CX': raise ValueError('Unsupported Clifford')
    c, t = gate['wires']
    return [state[j ^ (1 << t) if j & (1 << c) else j] for j in range(len(state))]


def expression(text, x):
    e = sp.sympify(text, rational=True)
    def visit(z):
        if z.is_Rational: return rational(z)
        if str(z) in ('a0', 'a1'): return x[int(str(z)[1])]
        if z.func == sp.Add: return sum((visit(a) for a in z.args), iv.mpf(0))
        if z.func == sp.Mul:
            out = iv.mpf(1)
            for a in z.args: out *= visit(a)
            return out
        if z.func == sp.Pow and z.args[1].is_Integer: return visit(z.args[0])**int(z.args[1])
        if z.func == sp.acos:
            a = visit(z.args[0]); return iv.atan2(iv.sqrt(1-a*a), a)
        if z.func == sp.atan: return iv.atan2(visit(z.args[0]), iv.mpf(1))
        raise ValueError('Unsupported data expression '+str(z))
    return visit(e)


def features(info, theta, x):
    n = info['n']; count = info['count']; size = 1 << n
    if len(theta) != count or len(x) != 2: raise ValueError('Witness dimension mismatch')
    if any(not -1 < F(str(v)) < 1 for v in x): raise ValueError('Input outside declared open domain')
    theta = [rational(v) for v in theta]; x = [rational(v) for v in x]
    enc = [expression(e, x) for e in info['data_expressions']]
    state = [iv.mpc(int(j == 0)) for j in range(size)]
    tangent = [[iv.mpc(0) for _ in range(size)] for _ in range(count)]
    for g in info['program']:
        if g['op'] != 'R':
            state = clifford(state, g); tangent = [clifford(v, g) for v in tangent]; continue
        angle = iv.mpf(0); factors = [iv.mpf(0) for _ in range(count)]
        for coef, ti, xi in g['terms']:
            a = rational(coef)*(enc[xi] if xi >= 0 else 1)
            angle += a*(theta[ti] if ti >= 0 else 1)
            if ti >= 0: factors[ti] += a
        c, s = iv.cos(angle/2), iv.sin(angle/2)
        ps = pauli(state, g['pauli'])
        for k in range(count):
            pt = pauli(tangent[k], g['pauli'])
            tangent[k] = [c*v-1j*s*w+factors[k]*(-s*a/2-1j*c*b/2)
                          for v, w, a, b in zip(tangent[k], pt, state, ps)]
        state = [c*a-1j*s*b for a, b in zip(state, ps)]
    f = []; jac = []
    for obs in info['observables']:
        ps = pauli(state, obs)
        # mpmath 1.3.0 iv.mpc.conjugate() has an internal tuple-shape bug.
        # Construct the conjugate explicitly from its enclosing real intervals.
        f.append(sum((iv.mpc(a.real, -a.imag)*b for a, b in zip(state, ps)), iv.mpc(0)).real)
        jac.append([2*sum((iv.mpc(a.real, -a.imag)*b for a, b in zip(v, ps)), iv.mpc(0)).real for v in tangent])
    return f, jac


def derivative(info, theta, x, xp):
    f, j = features(info, theta, x); fp, jp = features(info, theta, xp)
    d = sum((rational(w)*(a-b)**2 for w, a, b in zip(info['weights'], f, fp)), iv.mpf(0))
    grad = [sum((2*rational(w)*(a-b)*(r[k]-s[k])
                 for w, a, b, r, s in zip(info['weights'], f, fp, j, jp)), iv.mpf(0))
            for k in range(info['count'])]
    return d, grad


def nonzero(v): return 0 not in v


def bounds(v):
    out = []
    for sign, man, exponent, bits in v._mpi_:
        if bits < 0: raise ValueError('Nonfinite enclosure')
        out.append(str(F((-1 if sign else 1)*man)*F(2)**exponent))
    return out


def mid(v):
    a, b = bounds(v); return float((F(a)+F(b))/2)


def determinant(matrix):
    n = len(matrix)
    if n == 0: return iv.mpf(1)
    m = [row[:] for row in matrix]; det = iv.mpf(1)
    if any(len(row) != n for row in m): raise ValueError('Not square')
    for k in range(n):
        pivot = next((j for j in range(k, n) if nonzero(m[j][k])), None)
        if pivot is None: return None
        if pivot != k: m[k], m[pivot] = m[pivot], m[k]; det = -det
        p = m[k][k]; det *= p
        for j in range(k+1, n):
            factor = m[j][k]/p
            for i in range(k+1, n): m[j][i] -= factor*m[k][i]
    return det


def info_hash(info):
    return hashlib.sha256(json.dumps(info, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_witness(info, record):
    if record['source_sha256'] != info_hash(info): raise ValueError('Source hash mismatch')
    if record['space'] != 'original_inputs': raise ValueError('Only feasible original inputs are accepted')
    iv.dps = record['dps']
    _, grad = derivative(info, record['theta'], record['x'], record['xprime'])
    k = record['parameter']
    if not 0 <= k < info['count'] or not nonzero(grad[k]): raise ValueError('Uncertified ACTIVE')
    return bounds(grad[k])
