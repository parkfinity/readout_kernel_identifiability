#!/usr/bin/env python3
"""Exact generic kernel-rank certificates for the Kyriienko-type circuit family (n = 2, 4, 6, 8).

Object.  Projected kernel k = exp(-gamma d^2) with
    d^2(x, x'; theta) = sum_j w_j (f_j(x) - f_j(x'))^2,   f_j = Tr[rho_theta(x) O_j],
unit weights, readouts local XYZ (3n words), complete Pauli weight <= 2, local Z (secondary).
r = generic rank of theta -> d^2(., .; theta) (rank of the differential, maximised over theta).

Source -> IR.  Vendored frozen sQUlearn (commit 0f6cb098..., copied byte-identically into
./core_copy from cross_family_resource_v1/core), KyriienkoEncodingCircuit(num_qubits=n) with
constructor defaults, generate_initial_parameters(num_features=2, seed=0), Qiskit transpile with
the frozen census settings (optimization_level=3, seed_transpiler=71, approximation_degree=1,
num_processes=1, census basis), adapter.adapt, pipeline.program(fuse_adjacent=True), readouts as
cross_family_resource_v1/survey.py.  This is the route that produced the frozen n = 2, 4 IR.

Certificates (no floating threshold is ever used as a certificate).
  LOWER  r >= m: an m x m minor of the exact rational Jacobian J[pair, k] = d/dtheta_k d^2(x, x')
         at an exactly specified feasible point is nonzero.  Exact Gaussian-rational statevector:
         theta_k = 4 arctan(u_k) (equivalently 2 arctan t_k with t_k = 2u_k/(1-u_k^2)), so every
         gate half-angle theta_k/2 = 2 arctan u_k has rational cos/sin; inputs x_b = cos(phi_b),
         phi_b = 2 arctan(s_b), and every data half-angle is an integer multiple of phi_b (checked).
         Rows are scaled by positive rationals to integers; the minor is selected by elimination
         modulo a prime and its integer determinant is computed exactly (Bareiss), re-checked by
         SymPy DomainMatrix over ZZ and by residues modulo three primes.  rank is lower
         semicontinuous, so the generic rank is >= m.
  UPPER  (a) manuscript manuscript_theory_reference/family_extension.tex Eq. (kyrifamilyupper):
         r <= 3n - 2 for isotropic local XYZ or complete weight <= k < n readout;
         (b) machine re-verification on the generated IR: exact iterative deletion of single-angle
         rotations whose output-transported generator P satisfies [Omega, P(x)I + I(x)P] = 0,
         Omega = sum_O w_O O (x) O (exact Pauli algebra), plus "virtual insertion" Euler fields
         (X or Y rotation inserted after a ZXZ Euler block; the 2x2 identity
         sum_k v_k dU/dtheta_k = -(i/2) sin(b) Q U is verified symbolically in SymPy).
         r <= K - rank(field family at theta*), valid by analyticity.
Numerical SVD ranks and random-point gauge checks are reported as NOT certificates.
"""
import argparse
import gzip
import hashlib
import inspect
import itertools
import json
import math
import os
import platform
import random
import resource
import sys

# Stored exact determinant certificates exceed Python's default digit limit.
# Inputs here are trusted local research artifacts, not untrusted network data.
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

import time
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORE = HERE / 'core_copy'
JOURNAL = HERE.parent
ORIGINAL_CORE = JOURNAL / 'cross_family_resource_v1' / 'core'
FROZEN = JOURNAL / 'certified_dimension_extension_v1' / 'frozen_inputs'
MANUSCRIPT = JOURNAL / 'manuscript_theory_reference'
MODULE = 'kyriienko_nonlinear_encoding_circuit'
READOUTS = ('local_xyz', 'pauli_weight_le_2', 'local_z')
COMPILE = dict(optimization_level=3, seed_transpiler=71, approximation_degree=1.0, num_processes=1)
PRIMES = (2 ** 61 - 1, 2 ** 62 - 57, 2 ** 63 - 25)
POINT_SEED = 20261003
# Validation gate: exact values closed for n = 2, 4 (fourier_gram_identifiability_v1, archived manuscript).
GATE_PRIMARY = {(2, 'local_xyz'): 4, (4, 'local_xyz'): 10, (4, 'pauli_weight_le_2'): 10}
GATE_SECONDARY = {(2, 'local_z'): 4, (4, 'local_z'): 8, (2, 'pauli_weight_le_2'): 0}
UPPER_REFERENCE = ('manuscript_theory_reference/family_extension.tex, subsection "An all-size '
                   'Kyriienko-type upper bound", Eq. (kyrifamilyupper): r <= 3n-2 for Eq. (kyriform) '
                   'with independent local Euler blocks, entangler commuting with Z_0 and X_{n-1}, '
                   'isotropic local XYZ or complete weight<=k (k<n) readout; see also fourier_gram.tex '
                   'Sec. "Closing the three Kyriienko intervals".')


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    return sha256_bytes(Path(p).read_bytes())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == 'darwin' else r / 2 ** 10


# ---------------------------------------------------------------------------------------------
# Source -> IR (frozen census route)
# ---------------------------------------------------------------------------------------------
def readout(n, kind):
    """Identical to cross_family_resource_v1/survey.py::readout (ordering fixes the IR hash)."""
    if kind == 'pauli_weight_le_2':
        obs = [''.join(p) for p in itertools.product('IXYZ', repeat=n)
               if 1 <= sum(c != 'I' for c in p) <= 2]
    elif kind in ('local_xyz', 'local_z'):
        obs = ['I' * q + a + 'I' * (n - q - 1)
               for a in ('Z' if kind == 'local_z' else 'XYZ') for q in range(n)]
    else:
        raise ValueError(kind)
    return obs, np.ones(len(obs)).tolist()


_SOURCE = {}


def _core_imports():
    if str(CORE) not in sys.path:
        sys.path.insert(0, str(CORE))
    import qiskit
    from qiskit import transpile
    from source_host import construct, inventory, verify_sources
    from adapter import adapt
    from pipeline import BASIS, program
    return qiskit, transpile, construct, inventory, verify_sources, adapt, BASIS, program


def source_ir(n):
    """Construct, compile and adapt the Kyriienko circuit exactly as the frozen census did."""
    if n in _SOURCE:
        return _SOURCE[n]
    qiskit, transpile, construct, inventory, verify_sources, adapt, BASIS, program = _core_imports()
    manifest = verify_sources()
    entry = {e['module']: e for e in inventory()}[MODULE]
    t0 = time.perf_counter()
    obj, qc, x, t, kwargs = construct(entry, n, 1)
    t1 = time.perf_counter()
    compiled = transpile(qc, basis_gates=BASIS, **COMPILE)
    t2 = time.perf_counter()
    gates, expr, meta = adapt(compiled, x, t)
    prog = json.loads(json.dumps(program(gates, fuse_adjacent=True)))
    t3 = time.perf_counter()
    out = dict(obj=obj, program=prog, expressions=list(expr), count=len(t), meta=dict(
        module=MODULE, entry=entry, constructor_kwargs=kwargs,
        signature=str(inspect.signature(type(obj).__init__)), get_params=repr(obj.get_params()),
        initial_parameters='generate_initial_parameters(num_features=2, seed=0)',
        vendored_commit=manifest['commit'], qiskit=qiskit.__version__,
        compile=dict(COMPILE, basis_gates=list(BASIS)), native_operations=meta['native_operations'],
        global_phase_ignored=meta['global_phase_ignored'],
        seconds=dict(construct=t1 - t0, transpile=t2 - t1, adapt=t3 - t2)))
    _SOURCE[n] = out
    return out


def make_info(n, kind):
    src = source_ir(n)
    obs, w = readout(n, kind)
    ids = sorted({int(i) for g in src['program'] if g['op'] == 'R'
                  for coef, i, j in g['terms'] if i >= 0 and coef != 0})
    return {'n': n, 'count': src['count'], 'program': src['program'], 'ids': ids,
            'data_expressions': src['expressions'], 'observables': obs, 'weights': w}


def ir_text(info):
    """Same serialisation as certified_dimension_extension_v1/experiment.py::save."""
    return json.dumps(info, indent=2, ensure_ascii=False) + '\n'


# ---------------------------------------------------------------------------------------------
# IR parsing (shared by the exact and float engines)
# ---------------------------------------------------------------------------------------------
BASES = {'acos(a0)': 0, 'acos(a1)': 1}


def parse_program(info):
    """-> list of ('R', word, {k: Fraction}, {b: Fraction}) or ('CX', c, t).

    An R gate is exp(-i alpha P / 2) with alpha = sum_k th[k] theta_k + sum_b da[b] acos(x_b).
    Pauli words are in ascending wire order (word[q] acts on qubit q = bit q).  Products
    theta*data and constant angles are rejected (outside this fixed-frequency engine)."""
    import sympy as sp
    slots = []
    for e in info['data_expressions']:
        c, base = sp.sympify(e, rational=True).as_coeff_Mul()
        if not c.is_Rational or str(base) not in BASES:
            raise ValueError('Unsupported data expression ' + e)
        slots.append((F(int(c.p), int(c.q)), BASES[str(base)]))
    gates = []
    for g in info['program']:
        if g['op'] == 'CX':
            c, t = g['wires']
            gates.append(('CX', int(c), int(t)))
        elif g['op'] == 'R':
            if len(g['pauli']) != info['n'] or set(g['pauli']) - set('IXYZ'):
                raise ValueError('Bad Pauli word')
            th, da = {}, {}
            for c, t, x in g['terms']:
                c = F(str(c))
                if t >= 0 and x >= 0:
                    raise ValueError('theta*data product term: not fixed-frequency')
                if t < 0 and x < 0:
                    raise ValueError('Constant angle term not supported')
                if t >= 0:
                    th[int(t)] = th.get(int(t), F(0)) + c
                else:
                    m, b = slots[x]
                    da[b] = da.get(b, F(0)) + c * m
            th = {k: v for k, v in th.items() if v}
            da = {k: v for k, v in da.items() if v}
            gates.append(('R', g['pauli'], th, da))
        else:
            raise ValueError('Unsupported gate ' + g['op'])
    return gates


def cx_perm(n, c, t):
    j = np.arange(1 << n)
    return np.where((j >> c) & 1, j ^ (1 << t), j)


# ---------------------------------------------------------------------------------------------
# Exact Gaussian-rational statevector engine
# ---------------------------------------------------------------------------------------------
def gunit(r):
    """(A, B, C), gcd 1, with (A + iB)/C = exp(2i arctan r) = (1 + i r)^2 / (1 + r^2)."""
    r = F(r)
    p, q = r.numerator, r.denominator
    A, B, C = q * q - p * p, 2 * p * q, p * p + q * q
    g = math.gcd(A, B, C)
    return A // g, B // g, C // g


def unit_pow(u, e):
    A, B, C = u
    if e < 0:
        B, e = -B, -e
    ra, rb = 1, 0
    for _ in range(e):
        ra, rb = ra * A - rb * B, ra * B + rb * A
    return ra, rb, C ** e


def unit_prod(units):
    A, B, C = 1, 0, 1
    for a, b, c in units:
        A, B, C = A * a - B * b, A * b + B * a, C * c
    g = math.gcd(A, B, C)
    A, B, C = A // g, B // g, C // g
    if A * A + B * B != C * C:
        raise AssertionError('Not a unit Gaussian rational')
    return A, B, C


def unit_cos_sin_full(u):
    """cos and sin of the full angle theta when u encodes exp(i theta / 2)."""
    A, B, C = u
    return F(A * A - B * B, C * C), F(2 * A * B, C * C)


def obj_zeros(shape):
    return np.zeros(shape, dtype=object)


class ExactEngine:
    """Exact statevector over Q(i) with a common integer denominator.

    psi = Psi / D and d psi / d theta_k = Phi_k / (2 D) with Gaussian-integer vectors Psi, Phi_k.
    Gate exp(-i alpha P/2) with exp(i alpha/2) = (A + iB)/C acts as (A - iB P)/C.
    Then f_O = F_O / D^2 with F_O = <Psi|O|Psi> and d f_O / d theta_k = G_Ok / D^2 with
    G_Ok = Re <O Psi | Phi_k>.  All integers, no rounding anywhere."""

    def __init__(self, info):
        self.n = int(info['n'])
        self.N = 1 << self.n
        self.K = int(info['count'])
        self.gates = parse_program(info)
        for g in self.gates:
            if g[0] != 'R':
                continue
            for k, c in g[2].items():
                if c.denominator != 1:
                    raise ValueError('Non-integer theta coefficient: half-angle not rational')
            for b, d in g[3].items():
                if (d / 2).denominator != 1:
                    raise ValueError('Data half-angle is not an integer multiple of phi_b')
        self._pt = {}
        self._cx = {}

    def ptable(self, word):
        if word not in self._pt:
            xm = sum(1 << q for q, a in enumerate(word) if a in 'XY')
            zm = sum(1 << q for q, a in enumerate(word) if a in 'YZ')
            src = np.arange(self.N) ^ xm
            sgn = np.array([1 - 2 * (bin(int(s) & zm).count('1') & 1) for s in src], dtype=object)
            self._pt[word] = (src, sgn, word.count('Y') % 4)
        return self._pt[word]

    def pauli(self, re, im, word):
        """(re + i im) -> P (re + i im), along the last axis."""
        src, sgn, ny = self.ptable(word)
        r, i = re[..., src] * sgn, im[..., src] * sgn
        if ny == 0:
            return r, i
        if ny == 1:
            return -i, r
        if ny == 2:
            return -r, -i
        return i, -r

    def perm(self, c, t):
        if (c, t) not in self._cx:
            self._cx[c, t] = cx_perm(self.n, c, t)
        return self._cx[c, t]

    def run(self, wtheta, wdata, tangents=True):
        N, K = self.N, self.K
        re, im = obj_zeros(N), obj_zeros(N)
        re[0] = 1
        tr, ti = obj_zeros((K, N)), obj_zeros((K, N))
        D = 1
        for g in self.gates:
            if g[0] == 'CX':
                p = self.perm(g[1], g[2])
                re, im = re[p], im[p]
                if tangents:
                    tr, ti = tr[:, p], ti[:, p]
                continue
            _, word, th, da = g
            units = [unit_pow(wtheta[k], int(c)) for k, c in th.items()]
            units += [unit_pow(wdata[b], int(d / 2)) for b, d in da.items()]
            A, B, C = unit_prod(units)
            pr, pi = self.pauli(re, im, word)
            re, im = A * re + B * pi, A * im - B * pr
            if tangents:
                qr, qi = self.pauli(tr, ti, word)
                tr, ti = A * tr + B * qi, A * ti - B * qr
                if th:
                    pr, pi = self.pauli(re, im, word)
                    for k, c in th.items():
                        c = int(c)
                        tr[k] = tr[k] + c * pi
                        ti[k] = ti[k] - c * pr
            D *= C
        return D, re, im, tr, ti

    def features(self, obs, re, im, tr, ti):
        Fv = obj_zeros(len(obs))
        G = obj_zeros((len(obs), self.K))
        for a, o in enumerate(obs):
            orr, oi = self.pauli(re, im, o)
            Fv[a] = np.dot(re, orr) + np.dot(im, oi)
            if np.dot(re, oi) - np.dot(im, orr) != 0:
                raise AssertionError('Non-real Pauli expectation')
            G[a] = tr.dot(orr) + ti.dot(oi)
        return Fv, G

    def point(self, theta_u, s, obs):
        for v in s:
            x_from_s(v)  # enforces s > 0: acos(x_b) = phi_b = 2 arctan(s_b)
        wtheta = [gunit(u) for u in theta_u]
        wdata = [gunit(v) for v in s]
        D, re, im, tr, ti = self.run(wtheta, wdata)
        norm = np.dot(re, re) + np.dot(im, im)
        if norm != D * D:
            raise AssertionError('Exact state not normalised')
        Fv, G = self.features(obs, re, im, tr, ti)
        return dict(D=D, F=Fv, G=G)


def theta_from_u(u):
    return 4.0 * math.atan(float(F(u)))


def x_from_s(s):
    """x = cos(phi), phi = 2 arctan(s).  s > 0 puts phi in (0, pi), so arccos(x) = phi exactly
    (principal branch) and x lies in the open source domain (-1, 1)."""
    s = F(s)
    if s <= 0:
        raise ValueError('s must be positive (arccos branch)')
    return (1 - s * s) / (1 + s * s)


def pair_rows(data, idx, weights, pairs):
    """Exact integer Jacobian rows.  True row = scale * integer row,
    d/dtheta_k d^2(x_i, x_j) = 2 sum_O w_O (F_i D_j^2 - F_j D_i^2)(G_i D_j^2 - G_j D_i^2) / (D_i D_j)^4."""
    w = np.array([int(F(str(v))) for v in weights], dtype=object)
    if any(F(str(v)) != int(F(str(v))) or F(str(v)) <= 0 for v in weights):
        raise ValueError('Only positive integer weights are supported')
    rows, scales, d2 = [], [], []
    for i, j in pairs:
        a, b = data[i], data[j]
        Di2, Dj2 = a['D'] ** 2, b['D'] ** 2
        dF = a['F'][idx] * Dj2 - b['F'][idx] * Di2
        dG = a['G'][idx] * Dj2 - b['G'][idx] * Di2
        row = [int(v) for v in np.dot(w * dF, dG)]
        den = (Di2 * Dj2) ** 2
        d2.append(F(int(np.dot(w * dF, dF)), den))
        g = math.gcd(*row) if any(row) else 0
        if g:
            rows.append([v // g for v in row])
            scales.append(F(2 * g, den))
        else:
            rows.append(row)
            scales.append(F(1))
    return rows, scales, d2


# ---------------------------------------------------------------------------------------------
# Exact linear algebra
# ---------------------------------------------------------------------------------------------
def select_minor_mod(rows, p, limit=None):
    """Incremental row echelon modulo p.  Returns (rank_p, rows, pivot_columns); the integer
    submatrix on these rows and columns is nonsingular modulo p (triangular pivot structure)."""
    basis, rsel, csel = [], [], []
    for ri, row in enumerate(rows):
        v = [x % p for x in row]
        for pc, b in basis:
            f = v[pc]
            if f:
                v = [(a - f * bb) % p for a, bb in zip(v, b)]
        nz = next((c for c, a in enumerate(v) if a), None)
        if nz is None:
            continue
        inv = pow(v[nz], -1, p)
        basis.append((nz, [a * inv % p for a in v]))
        rsel.append(ri)
        csel.append(nz)
        if limit is not None and len(rsel) == limit:
            break
    return len(rsel), rsel, csel


def det_mod(M, p):
    A = [[x % p for x in r] for r in M]
    n, det = len(A), 1
    for k in range(n):
        piv = next((i for i in range(k, n) if A[i][k]), None)
        if piv is None:
            return 0
        if piv != k:
            A[k], A[piv] = A[piv], A[k]
            det = -det
        det = det * A[k][k] % p
        inv = pow(A[k][k], -1, p)
        for i in range(k + 1, n):
            f = A[i][k] * inv % p
            if f:
                A[i] = [(a - f * b) % p for a, b in zip(A[i], A[k])]
    return det % p


def bareiss_det(M):
    """Exact integer determinant (fraction-free Bareiss elimination)."""
    A = [list(map(int, r)) for r in M]
    n = len(A)
    if n == 0:
        return 1
    sign, prev = 1, 1
    for k in range(n - 1):
        if A[k][k] == 0:
            piv = next((i for i in range(k + 1, n) if A[i][k]), None)
            if piv is None:
                return 0
            A[k], A[piv] = A[piv], A[k]
            sign = -sign
        akk = A[k][k]
        for i in range(k + 1, n):
            aik = A[i][k]
            Ai, Ak = A[i], A[k]
            for j in range(k + 1, n):
                Ai[j] = (Ai[j] * akk - aik * Ak[j]) // prev
        prev = akk
    return sign * A[n - 1][n - 1]


def sympy_det(M):
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.domains import ZZ
    m = len(M)
    return int(DomainMatrix([[ZZ(int(v)) for v in r] for r in M], (m, m), ZZ).det())


def rank_exact_fraction(vectors, K):
    """Exact rank over Q of a few rational vectors (small families only)."""
    rows = [[F(v) for v in vec] for vec in vectors]
    rank, col = 0, 0
    for col in range(K):
        piv = next((i for i in range(rank, len(rows)) if rows[i][col] != 0), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(len(rows)):
            if i != rank and rows[i][col] != 0:
                f = rows[i][col] / rows[rank][col]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[rank])]
        rank += 1
    return rank


# ---------------------------------------------------------------------------------------------
# Exact Pauli algebra: transport, two-copy invariance, deletions, virtual Euler fields
# ---------------------------------------------------------------------------------------------
_PT = {('X', 'Y'): (1, 'Z'), ('Y', 'X'): (3, 'Z'), ('Y', 'Z'): (1, 'X'), ('Z', 'Y'): (3, 'X'),
       ('Z', 'X'): (1, 'Y'), ('X', 'Z'): (3, 'Y')}


def pauli_product(a, b):
    """a b = i^k w for Pauli words a, b."""
    k, out = 0, []
    for x, y in zip(a, b):
        if x == 'I':
            out.append(y)
        elif y == 'I':
            out.append(x)
        elif x == y:
            out.append('I')
        else:
            e, z = _PT[x, y]
            k += e
            out.append(z)
    return k % 4, ''.join(out)


def commutes(a, b):
    return sum(x != 'I' and y != 'I' and x != y for x, y in zip(a, b)) % 2 == 0


def cx_conj_word(word, c, t):
    """Word of CX P CX (sign dropped; signs do not affect commutation or two-copy invariance)."""
    code = {'I': (0, 0), 'X': (1, 0), 'Z': (0, 1), 'Y': (1, 1)}
    back = {v: k for k, v in code.items()}
    xs = [code[a][0] for a in word]
    zs = [code[a][1] for a in word]
    xs[t] ^= xs[c]
    zs[c] ^= zs[t]
    return ''.join(back[(x, z)] for x, z in zip(xs, zs))


def two_copy_invariant(obs, weights, P):
    """Exact test of [Omega, P(x)I + I(x)P] = 0, Omega = sum_O w_O O (x) O.
    [O(x)O, P(x)I + I(x)P] = [O,P](x)O + O(x)[O,P] and [O,P] = 2 O P for anticommuting O, P."""
    acc = {}
    unit = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
    for O, w in zip(obs, weights):
        if commutes(O, P):
            continue
        k, OP = pauli_product(O, P)
        re, im = unit[k]
        w = F(str(w))
        for key in ((OP, O), (O, OP)):
            a, b = acc.get(key, (F(0), F(0)))
            acc[key] = (a + 2 * w * re, b + 2 * w * im)
    return all(v == (0, 0) for v in acc.values())


def transport(later, word):
    """Move exp(-i t P/2) to the output through later gates: CX conjugates P; a later rotation
    must commute with the transported P (otherwise the sufficient rule fails -> None)."""
    P = word
    for g in later:
        if g[0] == 'CX':
            P = cx_conj_word(P, g[1], g[2])
        elif not commutes(P, g[1]):
            return None
    return P


def coordinate_deletions(info):
    """Iterative exact deletion of single-angle rotations whose output generator is a two-copy
    symmetry of the readout metric.  Each deleted angle is globally kernel-inactive (for every
    theta and every input pair); deleting it does not change the kernel of the remaining angles."""
    gates = parse_program(info)
    obs, w = info['observables'], info['weights']
    occurrences = {}
    for i, g in enumerate(gates):
        if g[0] == 'R':
            for k in g[2]:
                occurrences.setdefault(k, []).append(i)
    active = list(range(len(gates)))
    deleted = []
    changed = True
    while changed:
        changed = False
        for idx in reversed(active):
            g = gates[idx]
            if g[0] != 'R' or g[3] or len(g[2]) != 1:
                continue
            k = next(iter(g[2]))
            if occurrences[k] != [idx]:
                continue
            P = transport([gates[j] for j in active if j > idx], g[1])
            if P is not None and two_copy_invariant(obs, w, P):
                deleted.append(dict(parameter=k, gate_index=idx, generator=g[1], output_generator=P))
                active.remove(idx)
                changed = True
                break
    return deleted


def euler_blocks(info):
    """Per qubit: the trailing (Z a, X b, Z c) single-angle rotations, consecutive on the wire."""
    gates = parse_program(info)
    n = info['n']
    blocks = []
    for q in range(n):
        wire = [(i, g) for i, g in enumerate(gates)
                if (g[0] == 'R' and g[1][q] != 'I') or (g[0] == 'CX' and q in g[1:])]
        rot = [(i, g) for i, g in wire if g[0] == 'R']
        last3 = rot[-3:]
        pos = [wire.index(e) for e in last3]
        ok = (len(last3) == 3 and pos == list(range(pos[0], pos[0] + 3))
              and all(set(g[1]) <= {'I', a} and g[1][q] == a and len(g[2]) == 1 and not g[3]
                      for (i, g), a in zip(last3, 'ZXZ'))
              and all(g[0] == 'CX' for _, g in wire[pos[-1] + 1:]))
        if not ok:
            blocks.append(None)
            continue
        params = tuple(next(iter(g[2])) for _, g in last3)
        coefs = tuple(next(iter(g[2].values())) for _, g in last3)
        blocks.append(dict(qubit=q, params=params, coefficients=[str(c) for c in coefs],
                           last_gate_index=last3[-1][0]))
    return blocks


def euler_field(axis, a, b, c, cs):
    """Cramer field (va, vb, vc) for U = Rz(c) Rx(b) Rz(a): sum v_k dU/dk = -(i/2) sin(b) Q U.
    cs(angle) -> (cos, sin)."""
    (ca, sa), (cb, sb), (cc, sc) = cs(a), cs(b), cs(c)
    if axis == 'X':
        return (sc, cc * sb, -sc * cb)
    if axis == 'Y':
        return (-cc, sc * sb, cc * cb)
    raise ValueError(axis)


def euler_identity_symbolic():
    """SymPy proof of the 2x2 identities behind the virtual fields (and a rejected control)."""
    import sympy as sp
    a, b, c = sp.symbols('a b c', real=True)
    X = sp.Matrix([[0, 1], [1, 0]])
    Y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    Z = sp.Matrix([[1, 0], [0, -1]])

    def R(P, t):
        return sp.cos(t / 2) * sp.eye(2) - sp.I * sp.sin(t / 2) * P
    U = R(Z, c) * R(X, b) * R(Z, a)
    cs = lambda t: (sp.cos(t), sp.sin(t))  # noqa: E731
    out = {}
    for axis, Q in (('X', X), ('Y', Y)):
        v = euler_field(axis, a, b, c, cs)
        for label, vv in (('field', v), ('control_doubled_first_component', (2 * v[0], v[1], v[2]))):
            lhs = vv[0] * U.diff(a) + vv[1] * U.diff(b) + vv[2] * U.diff(c)
            rhs = -sp.I / 2 * sp.sin(b) * Q * U
            resid = (lhs - rhs).applyfunc(lambda e: sp.simplify(sp.expand(sp.expand_trig(e).rewrite(sp.exp))))
            out[axis + '_' + label + '_is_identity'] = bool(all(e == 0 for e in resid))
    out['passed'] = (out['X_field_is_identity'] and out['Y_field_is_identity']
                     and not out['X_control_doubled_first_component_is_identity']
                     and not out['Y_control_doubled_first_component_is_identity'])
    out['statement'] = ('U(a,b,c)=Rz(c)Rx(b)Rz(a); X: v=(sin c, cos c sin b, -sin c cos b); '
                        'Y: v=(-cos c, sin c sin b, cos c cos b); sum_k v_k dU/dk = -(i/2) sin(b) Q U')
    return out


def virtual_fields(info):
    """Virtual insertion of exp(-i e Q/2) (Q in X, Y) right after each ZXZ Euler block; accepted
    if the transported Q is a two-copy symmetry of the readout metric (exact)."""
    gates = parse_program(info)
    n, obs, w = info['n'], info['observables'], info['weights']
    out = []
    for blk in euler_blocks(info):
        if blk is None or any(F(c) != 1 for c in blk['coefficients']):
            continue
        q = blk['qubit']
        for axis in ('X', 'Y'):
            word = 'I' * q + axis + 'I' * (n - q - 1)
            P = transport(gates[blk['last_gate_index'] + 1:], word)
            if P is not None and two_copy_invariant(obs, w, P):
                out.append(dict(qubit=q, axis=axis, params=list(blk['params']), output_generator=P))
    return out


def field_family(K, deleted, vfields, cs):
    vecs = []
    for d in deleted:
        v = [0] * K
        v[d['parameter']] = 1
        vecs.append(v)
    for f in vfields:
        a, b, c = f['params']
        v = [0] * K
        for k, val in zip((a, b, c), euler_field(f['axis'], a, b, c, cs)):
            v[k] = val
        vecs.append(v)
    return vecs


def upper_bound_certificate(info, theta_u):
    K = info['count']
    t0 = time.perf_counter()
    deleted = coordinate_deletions(info)
    vf = virtual_fields(info)
    units = [gunit(u) for u in theta_u]
    vecs = field_family(K, deleted, vf, lambda k: unit_cos_sin_full(units[k]))
    m = rank_exact_fraction(vecs, K)
    return dict(deleted_coordinates=deleted, virtual_fields=vf, family_rank_at_theta_star=m,
                upper=K - m, exact_null_vectors_at_theta_star=[[str(F(v)) for v in vec] for vec in vecs],
                seconds=time.perf_counter() - t0,
                logic=('Each deleted angle is globally inactive (transported generator P with '
                       '[Omega,P(x)I+I(x)P]=0, exact Pauli algebra). Each virtual field v(theta) obeys '
                       'J(theta)v(theta)=0 for all theta (SymPy Euler identity + exact two-copy '
                       'invariance of the transported inserted generator). Family rank m at theta* is '
                       '<= its generic rank, so generic r <= K-m (analyticity).'))


# ---------------------------------------------------------------------------------------------
# Structural checks of the generated IR against the source formula and Eq. (kyriform)
# ---------------------------------------------------------------------------------------------
def wire_sequences_from_info(info):
    gates = parse_program(info)
    n = info['n']
    seq = [[] for _ in range(n)]
    for g in gates:
        if g[0] == 'CX':
            for q in g[1:]:
                seq[q].append(('CX', g[1], g[2]))
        else:
            sup = [q for q, a in enumerate(g[1]) if a != 'I']
            if len(sup) != 1:
                for q in sup:
                    seq[q].append(('R', g[1], tuple(sorted(g[2].items())), tuple(sorted(g[3].items()))))
                continue
            q = sup[0]
            ang = {('theta', k): v for k, v in g[2].items()}
            ang.update({('acos', b): v for b, v in g[3].items()})
            if seq[q] and seq[q][-1][0] == 'R1' and seq[q][-1][1] == g[1][q]:
                prev = dict(seq[q][-1][2])
                for key, v in ang.items():
                    prev[key] = prev.get(key, F(0)) + v
                seq[q][-1] = ('R1', g[1][q], tuple(sorted((k, v) for k, v in prev.items() if v)))
            else:
                seq[q].append(('R1', g[1][q], tuple(sorted(ang.items()))))
    return seq


def wire_sequences_from_formula(n, num_features=2):
    """Expected circuit straight from the vendored source formula (no Qiskit, no IR):
    KyriienkoEncodingCircuit defaults (chebyshev_tower, HEA, 1 encoding + 1 variational layer,
    rotation ry, alpha = 2, num_chebyshev = n).  Encoding gate m = 0..2n-1 is
    ry(2 (m//2 + 1) arccos x_{m mod 2}) on qubit m mod n; then RZ RX RZ per qubit with
    theta_{3q}, theta_{3q+1}, theta_{3q+2}; then cx(i, i+1) for even i, then for odd i."""
    seq = [[] for _ in range(n)]

    def add_rot(q, axis, ang):
        if seq[q] and seq[q][-1][0] == 'R1' and seq[q][-1][1] == axis:
            prev = dict(seq[q][-1][2])
            for key, v in ang.items():
                prev[key] = prev.get(key, F(0)) + v
            seq[q][-1] = ('R1', axis, tuple(sorted((k, v) for k, v in prev.items() if v)))
        else:
            seq[q].append(('R1', axis, tuple(sorted(ang.items()))))
    for m in range(num_features * n):
        add_rot(m % n, 'Y', {('acos', m % num_features): F(2 * (m // num_features + 1))})
    for q in range(n):
        for j, axis in enumerate('ZXZ'):
            add_rot(q, axis, {('theta', 3 * q + j): F(1)})
    for start in (0, 1):
        for i in range(start, n - 1, 2):
            seq[i].append(('CX', i, i + 1))
            seq[i + 1].append(('CX', i, i + 1))
    return seq


def kappa_table(n):
    """kappa_q and feature index per qubit from the source formula."""
    out = []
    for q in range(n):
        ms = [m for m in range(2 * n) if m % n == q]
        out.append(dict(qubit=q, feature=sorted({m % 2 for m in ms}),
                        kappa=sum(2 * (m // 2 + 1) for m in ms)))
    return out


def entangler_commutes(info, word):
    """Exact check E P E^dag = P for the CX entangler (the IR's CX gates, in IR order)."""
    gates = parse_program(info)
    n = info['n']
    j = np.arange(1 << n)
    pi = j.copy()
    for g in gates:
        if g[0] == 'CX':
            c, t = g[1], g[2]
            pi = pi ^ (((pi >> c) & 1) << t)
    xm = sum(1 << q for q, a in enumerate(word) if a in 'XY')
    zm = sum(1 << q for q, a in enumerate(word) if a in 'YZ')
    par = lambda v: np.array([bin(int(s) & zm).count('1') & 1 for s in v])  # noqa: E731
    return bool(np.array_equal(pi[j ^ xm], pi ^ xm) and np.array_equal(par(j), par(pi)))


def structure_checks(info):
    n = info['n']
    gates = parse_program(info)
    formula = wire_sequences_from_formula(n)
    ir = wire_sequences_from_info(info)
    params = [k for g in gates if g[0] == 'R' for k in g[2]]
    blocks = euler_blocks(info)
    fixed_frequency = all(not (g[2] and g[3]) for g in gates if g[0] == 'R')
    kyri = (all(b is not None and b['params'] == (3 * q, 3 * q + 1, 3 * q + 2) for q, b in enumerate(blocks))
            and sorted(params) == list(range(info['count'])) and info['count'] == 3 * n)
    obs, w = info['observables'], info['weights']
    return dict(
        formula_equivalent=formula == ir,
        kappa=kappa_table(n),
        fixed_frequency=fixed_frequency,
        kyriform_euler_blocks=kyri,
        each_parameter_once=len(params) == len(set(params)),
        entangler=[[g[1], g[2]] for g in gates if g[0] == 'CX'],
        entangler_commutes_Z0=entangler_commutes(info, 'Z' + 'I' * (n - 1)),
        entangler_commutes_X_last=entangler_commutes(info, 'I' * (n - 1) + 'X'),
        unit_weights=all(F(str(v)) == 1 for v in w),
        readout_words=len(obs))


# ---------------------------------------------------------------------------------------------
# Float engines (diagnostics, cross-checks and numerical ranks; never certificates)
# ---------------------------------------------------------------------------------------------
class FloatEngine:
    """complex128 statevector from the IR, vectorised over input points, analytic tangents."""

    def __init__(self, info):
        self.n = int(info['n'])
        self.N = 1 << self.n
        self.K = int(info['count'])
        self.gates = parse_program(info)
        self._pt = {}

    def ptable(self, word):
        if word not in self._pt:
            xm = sum(1 << q for q, a in enumerate(word) if a in 'XY')
            zm = sum(1 << q for q, a in enumerate(word) if a in 'YZ')
            src = np.arange(self.N) ^ xm
            par = np.array([bin(int(s) & zm).count('1') & 1 for s in src])
            ph = (1j) ** (word.count('Y') % 4) * (1 - 2 * par)
            self._pt[word] = (src, ph)
        return self._pt[word]

    def pauli(self, s, word):
        src, ph = self.ptable(word)
        return s[..., src] * ph

    def simulate(self, theta, phi, obs, tangents=True):
        """phi: (P, 2) with phi_b = arccos(x_b).  Returns f (|O|, P) and df (K, |O|, P)."""
        P = phi.shape[0]
        s = np.zeros((P, self.N), complex)
        s[:, 0] = 1
        t = np.zeros((self.K, P, self.N), complex) if tangents else None
        for g in self.gates:
            if g[0] == 'CX':
                p = cx_perm(self.n, g[1], g[2])
                s = s[..., p]
                if tangents:
                    t = t[..., p]
                continue
            _, word, th, da = g
            ang = np.zeros(P)
            for k, c in th.items():
                ang = ang + float(c) * theta[k]
            for b, d in da.items():
                ang = ang + float(d) * phi[:, b]
            co, si = np.cos(ang / 2)[:, None], np.sin(ang / 2)[:, None]
            s_new = co * s - 1j * si * self.pauli(s, word)
            if tangents:
                t = co * t - 1j * si * self.pauli(t, word)
                ps = self.pauli(s_new, word)
                for k, c in th.items():
                    t[k] += -0.5j * float(c) * ps
            s = s_new
        f = np.empty((len(obs), P))
        df = np.empty((self.K, len(obs), P)) if tangents else None
        for a, o in enumerate(obs):
            po = self.pauli(s, o)
            f[a] = np.real(np.sum(s.conj() * po, -1))
            if tangents:
                df[:, a] = 2 * np.real(np.sum(t.conj() * po[None], -1))
        return f, df

    def jacobian(self, theta, X, Xp, obs, weights):
        """Rows: pairs (X[i], Xp[i]); returns d2 (P,) and J (P, K)."""
        w = np.array([float(v) for v in weights])
        phi = np.arccos(np.concatenate([X, Xp]))
        f, df = self.simulate(theta, phi, obs)
        P = len(X)
        Df = f[:, :P] - f[:, P:]
        Ddf = df[:, :, :P] - df[:, :, P:]
        d2 = np.sum(w[:, None] * Df ** 2, 0)
        J = 2 * np.einsum('o,op,kop->pk', w, Df, Ddf)
        return d2, J


def dense_source_state(n, theta, x):
    """Independent dense statevector straight from the source formula (no IR, no Qiskit):
    tensor contractions with Qiskit gate conventions; qubit q <-> bit q (axis n-1-q)."""
    psi = np.zeros((2,) * n, complex)
    psi[(0,) * n] = 1

    def apply1(psi, U, q):
        ax = n - 1 - q
        return np.moveaxis(np.tensordot(U, psi, axes=([1], [ax])), 0, ax)

    def ry(t):
        c, s = math.cos(t / 2), math.sin(t / 2)
        return np.array([[c, -s], [s, c]], complex)

    def rx(t):
        c, s = math.cos(t / 2), math.sin(t / 2)
        return np.array([[c, -1j * s], [-1j * s, c]])

    def rz(t):
        return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])

    def cx(psi, c, t):
        psi = psi.copy()
        axc, axt = n - 1 - c, n - 1 - t
        idx = [slice(None)] * n
        idx[axc] = 1
        sub = psi[tuple(idx)]
        psi[tuple(idx)] = np.flip(sub, axis=axt - (1 if axt > axc else 0))
        return psi
    for m in range(2 * n):
        psi = apply1(psi, ry(2.0 * (m // 2 + 1) * math.acos(x[m % 2])), m % n)
    for q in range(n):
        psi = apply1(psi, rz(theta[3 * q]), q)
        psi = apply1(psi, rx(theta[3 * q + 1]), q)
        psi = apply1(psi, rz(theta[3 * q + 2]), q)
    for start in (0, 1):
        for i in range(start, n - 1, 2):
            psi = cx(psi, i, i + 1)
    return psi


def dense_expectations(n, psi, obs):
    mats = {'X': np.array([[0, 1], [1, 0]], complex), 'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]], complex)}
    out = []
    for o in obs:
        phi = psi
        for q, a in enumerate(o):
            if a != 'I':
                ax = n - 1 - q
                phi = np.moveaxis(np.tensordot(mats[a], phi, axes=([1], [ax])), 0, ax)
        out.append(float(np.real(np.vdot(psi, phi))))
    return np.array(out)


def dense_d2(n, theta, x, xp, obs, weights):
    fa = dense_expectations(n, dense_source_state(n, theta, x), obs)
    fb = dense_expectations(n, dense_source_state(n, theta, xp), obs)
    return float(np.sum(np.array([float(v) for v in weights]) * (fa - fb) ** 2)), fa, fb


def qiskit_source_features(n, theta, X, obs):
    """Features from the vendored source object via Qiskit Statevector (core_copy/adapter.py)."""
    _core_imports()
    from adapter import source_states, source_features
    obj = source_ir(n)['obj']
    return source_features(source_states(obj, np.asarray(X, float), np.asarray(theta, float)), obs)


# ---------------------------------------------------------------------------------------------
# Numerical checks (NOT certificates)
# ---------------------------------------------------------------------------------------------
def numerical_checks(info, upper_info, seeds=(11, 12, 13, 14), tol=1e-9):
    eng = FloatEngine(info)
    K = info['count']
    obs, w = info['observables'], info['weights']
    trials = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        theta = rng.uniform(-np.pi, np.pi, K)
        P = max(3 * K, 60)
        X = rng.uniform(-1, 1, (P, 2))
        Xp = rng.uniform(-1, 1, (P, 2))
        d2, J = eng.jacobian(theta, X, Xp, obs, w)
        sv = np.linalg.svd(J, compute_uv=False)
        # Reference scale: sigma_max, but never below the size of the distances themselves, so an
        # identically-zero Jacobian (e.g. complete Pauli readout) is not rescaled into noise.
        ref = max(float(sv[0]) if len(sv) else 0.0, float(np.max(np.abs(d2))), 1e-300)
        r = int(np.sum(sv > tol * ref))
        vecs = field_family(K, upper_info['deleted_coordinates'], upper_info['virtual_fields'],
                            lambda k: (math.cos(theta[k]), math.sin(theta[k])))
        nJ = max(float(np.linalg.norm(J)), float(np.max(np.abs(d2))), 1e-300)
        gauge = []
        labels = ([('deleted', d['parameter']) for d in upper_info['deleted_coordinates']]
                  + [('virtual', f['qubit'], f['axis']) for f in upper_info['virtual_fields']])
        for v, lab in zip(vecs, labels):
            v = np.array([float(a) for a in v])
            res = float(np.linalg.norm(J @ v) / (nJ * np.linalg.norm(v)))
            ctrl = v.copy()
            if lab[0] == 'deleted':   # control: move weight onto the neighbouring coordinate
                ctrl[(int(lab[1]) + K - 1) % K] += 1.0
                ctrl_kind = 'e_k + e_(k-1)'
            else:                     # control: double the first nonzero component
                k0 = int(np.flatnonzero(v)[0])
                ctrl[k0] *= 2.0
                ctrl_kind = 'first component doubled'
            cres = float(np.linalg.norm(J @ ctrl) / (nJ * np.linalg.norm(ctrl)))
            gauge.append(dict(field=list(map(str, lab)), relative_residual=res,
                              control=ctrl_kind, control_relative_residual=cres))
        trials.append(dict(seed=sd, pairs=P, rank=r, sigma_max=float(sv[0]) if len(sv) else 0.0,
                           reference_scale=ref,
                           sigma_r_rel=float(sv[r - 1] / ref) if r else None,
                           sigma_r1_rel=float(sv[r] / ref) if r < len(sv) else None,
                           singular_values_rel=[float(v / ref) for v in sv],
                           gauge_fields=gauge))
    return dict(certified=False, label='NOT a certificate (floating SVD / random-point residuals)',
                tolerance_relative=tol, trials=trials,
                rank=max(t['rank'] for t in trials), min_rank=min(t['rank'] for t in trials),
                max_gauge_relative_residual=max([g['relative_residual'] for t in trials
                                                 for g in t['gauge_fields']] or [0.0]),
                min_control_relative_residual=min([g['control_relative_residual']
                                                   for t in trials for g in t['gauge_fields']] or [0.0]),
                note=('relative residual ||J v|| / (max(||J||_F, max d^2) ||v||); controls are perturbed '
                      'fields that should NOT be null (they are null only when J itself vanishes)'))


# ---------------------------------------------------------------------------------------------
# Exact point specification
# ---------------------------------------------------------------------------------------------
S_CHOICES = [F(1, 2), F(2), F(1, 3), F(3), F(2, 3), F(3, 2), F(1, 4), F(4), F(3, 4), F(4, 3),
             F(1, 5), F(5), F(2, 5), F(5, 2), F(3, 5), F(5, 3), F(4, 5), F(5, 4), F(1, 6), F(6)]


def theta_star(K, attempt=0):
    """u_k for theta_k = 4 arctan(u_k); attempt 0 uses u_k = (k+2)/(k+5) (cf. manuscript t_j)."""
    if attempt == 0:
        return [F(k + 2, k + 5) for k in range(K)]
    rng = random.Random(POINT_SEED + 1000 * attempt)
    return [F(rng.randint(1, 9), rng.randint(10, 19)) for _ in range(K)]


def input_points(P, seed=POINT_SEED):
    rng = random.Random(seed)
    cand = [(a, b) for a in S_CHOICES for b in S_CHOICES if a != b]
    return rng.sample(cand, P)


def point_record(s):
    return dict(s=[str(v) for v in s], x=[str(x_from_s(v)) for v in s],
                formula='x_b = cos(phi_b), phi_b = 2*arctan(s_b) in (0, pi); x_b = (1-s_b^2)/(1+s_b^2)')


class PointCache:
    """Exact states/features per (n, theta attempt, input point), for the union readout."""

    def __init__(self, info_union):
        self.info = info_union
        self.engine = ExactEngine(info_union)
        self.obs = info_union['observables']
        self.index = {o: i for i, o in enumerate(self.obs)}
        self.data = {}
        self.seconds = 0.0

    def get(self, theta_u, pts):
        out = []
        for s in pts:
            key = (tuple(theta_u), tuple(s))
            if key not in self.data:
                t0 = time.perf_counter()
                self.data[key] = self.engine.point(theta_u, s, self.obs)
                self.seconds += time.perf_counter() - t0
            out.append(self.data[key])
        return out


# ---------------------------------------------------------------------------------------------
# Certification of one setting
# ---------------------------------------------------------------------------------------------
def manuscript_upper(n, kind):
    if kind == 'local_xyz' or (kind == 'pauli_weight_le_2' and n > 2):
        return dict(manuscript_value=3 * n - 2, manuscript_reference=UPPER_REFERENCE)
    if kind == 'pauli_weight_le_2' and n == 2:
        return dict(manuscript_value=0, manuscript_reference=(
            'family_extension.tex (k = n: all final variational parameters invisible to full Pauli '
            'distance, r = 0) and fourier_gram.tex Prop. IC'))
    return dict(manuscript_value=None, manuscript_reference=(
        'none in the manuscript for local Z at this size; upper bound is the machine certificate '
        '(exact deletion of the n trailing Rz angles, whose output generators are Z strings '
        'commuting with every Z readout)'))


def certify_setting(info, cache, n_points=10, out_dir=None, name=None, numeric=True, log=print):
    n, K = info['n'], info['count']
    timings = {}
    t_all = time.perf_counter()
    obs, w = info['observables'], info['weights']
    if (info['program'] != cache.info['program'] or info['data_expressions'] != cache.info['data_expressions']
            or info['count'] != cache.info['count']):
        raise AssertionError('Cached exact states belong to a different circuit')
    idx = np.array([cache.index[o] for o in obs])
    t0 = time.perf_counter()
    struct = structure_checks(info)
    timings['structure_checks'] = time.perf_counter() - t0

    attempts = []
    chosen = None
    for attempt in range(3):
        theta_u = theta_star(K, attempt)
        t0 = time.perf_counter()
        ub = upper_bound_certificate(info, theta_u)
        timings['upper_bound_certificate'] = time.perf_counter() - t0
        P = n_points
        for grow in range(3):
            pts = input_points(P)
            pairs = list(itertools.combinations(range(P), 2))
            t0 = time.perf_counter()
            before = cache.seconds
            data = cache.get(theta_u, pts)
            t_states = cache.seconds - before
            rows, scales, d2 = pair_rows(data, idx, w, pairs)
            t_rows = time.perf_counter() - t0 - t_states
            t0 = time.perf_counter()
            rank_p, rsel, csel = select_minor_mod(rows, PRIMES[0])
            t_sel = time.perf_counter() - t0
            attempts.append(dict(theta_attempt=attempt, points=P, pairs=len(pairs), rank_mod_p=rank_p,
                                 field_upper=ub['upper'], exact_state_seconds=t_states,
                                 rows_seconds=t_rows, selection_seconds=t_sel))
            log('  n=%d %s attempt=%d P=%d rank_mod_p=%d upper=%d (states %.1fs rows %.1fs)'
                % (n, info.get('_kind', ''), attempt, P, rank_p, ub['upper'], t_states, t_rows))
            if rank_p >= ub['upper']:
                break
            P += 6
        chosen = (attempt, theta_u, ub, P, pts, pairs, rows, scales, d2, rank_p, rsel, csel)
        if rank_p >= ub['upper']:
            break
    attempt, theta_u, ub, P, pts, pairs, rows, scales, d2, rank_p, rsel, csel = chosen
    timings['exact_states_total_cache'] = cache.seconds

    # Exact determinant of the selected minor.
    order = rank_p
    rs, cs_ = sorted(rsel), sorted(csel)
    M = [[rows[i][k] for k in cs_] for i in rs]
    t0 = time.perf_counter()
    det_int = bareiss_det(M) if order else 1
    timings['bareiss_det'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    det_sym = sympy_det(M) if order else 1
    timings['sympy_det'] = time.perf_counter() - t0
    residues = {str(p): dict(det_mod_p=det_mod(M, p) if order else 1, bareiss_mod_p=det_int % p)
                for p in PRIMES}
    scale = F(1)
    for i in rs:
        scale *= scales[i]
    det_rat = scale * det_int
    lower = order if det_int != 0 and det_sym == det_int else None
    if lower is None:
        raise AssertionError('Exact determinant failed to confirm the modular minor')

    # Exact consistency at theta*: the exact null vectors annihilate every exact row.
    t0 = time.perf_counter()
    null_ok = []
    for vec in ub['exact_null_vectors_at_theta_star']:
        v = [F(a) for a in vec]
        null_ok.append(all(sum(F(r[k]) * v[k] for k in range(K) if v[k]) == 0 for r in rows))
    timings['exact_null_vector_check'] = time.perf_counter() - t0
    exact_rank_at_point = order if (all(null_ok) and order == ub['upper']) else None

    # Cross-checks of the exact engine at the exact point.
    t0 = time.perf_counter()
    theta_f = np.array([theta_from_u(u) for u in theta_u])
    X = np.array([[float(x_from_s(v)) for v in s] for s in pts])
    feng = FloatEngine(info)
    ffl, _ = feng.simulate(theta_f, np.arccos(X), obs, tangents=False)
    fex = np.array([[float(F(int(dd['F'][i]), dd['D'] ** 2)) for i in idx] for dd in data]).T
    err_float = float(np.max(np.abs(ffl - fex)))
    k_dense = min(3, P)
    fden = np.array([dense_expectations(n, dense_source_state(n, theta_f, X[m]), obs) for m in range(k_dense)]).T
    err_dense = float(np.max(np.abs(fden - fex[:, :k_dense])))
    try:
        fq = qiskit_source_features(n, theta_f, X[:k_dense], obs).T
        err_qiskit = float(np.max(np.abs(fq - fex[:, :k_dense])))
    except Exception as e:  # pragma: no cover - environment dependent
        err_qiskit = 'unavailable: %r' % (e,)
    # exact derivative vs float tangent (same point)
    _, dfl = feng.simulate(theta_f, np.arccos(X[:2]), obs)
    gex = np.array([[[float(F(int(dd['G'][i][k]), dd['D'] ** 2)) for k in range(K)] for i in idx] for dd in data[:2]])
    err_grad = float(np.max(np.abs(np.transpose(dfl, (2, 1, 0)) - gex)))
    timings['cross_checks'] = time.perf_counter() - t0

    num = None
    if numeric:
        t0 = time.perf_counter()
        num = numerical_checks(info, ub)
        timings['numerical_checks'] = time.perf_counter() - t0

    certified = lower == ub['upper']
    digits = len(str(abs(det_int)))
    jac_file = None
    if out_dir is not None:
        payload = dict(setting=name, theta_u=[str(u) for u in theta_u], points=[point_record(s) for s in pts],
                       pairs=pairs, row_scales=[str(s) for s in scales], integer_rows=[[str(v) for v in r] for r in rows],
                       note='J[pair,k] = row_scale * integer_row[k] = d/dtheta_k d^2(x_i, x_j) exactly')
        raw = json.dumps(payload).encode()
        jac_file = name + '_jacobian.json.gz'
        with open(Path(out_dir) / jac_file, 'wb') as fh:
            fh.write(gzip.compress(raw, mtime=0))
        jac_sha = sha256_bytes(raw)
    else:
        jac_sha = sha256_bytes(json.dumps([[str(v) for v in r] for r in rows]).encode())
    timings['total'] = time.perf_counter() - t_all
    res = dict(
        setting=name, family=MODULE, n=n, readout=info.get('_kind'), K=K, conjecture_3n_minus_2=3 * n - 2,
        observables=len(obs), ir_sha256=sha256_bytes(ir_text({k: v for k, v in info.items() if not k.startswith('_')}).encode()),
        structure=struct,
        certified_lower_bound=dict(
            value=lower, method='nonzero exact minor of the d^2 Jacobian at an exact feasible point',
            theta_point=dict(u=[str(u) for u in theta_u], attempt=attempt,
                             formula=('theta_k = 4*arctan(u_k) = 2*arctan(t_k), t_k = 2u_k/(1-u_k^2); '
                                      'exp(i theta_k/2) = (1+i u_k)^2/(1+u_k^2) is a Gaussian rational')),
            input_points=[point_record(s) for s in pts], pairs=[list(p) for p in pairs],
            minor_rows_pair_indices=rs, minor_rows_pairs=[list(pairs[i]) for i in rs], minor_columns_theta=cs_,
            excluded_columns_theta=[k for k in range(K) if k not in cs_],
            integer_minor_determinant=str(det_int), integer_minor_digits=digits,
            row_scale_product=str(scale),
            rational_minor_determinant=str(det_rat),
            rational_minor_log10_abs=(math.log10(abs(det_rat.numerator)) - math.log10(det_rat.denominator)) if det_rat else None,
            sympy_domainmatrix_det_equal=det_sym == det_int, residues=residues,
            selection='row echelon modulo p=2^61-1 (exact integer arithmetic); determinant exact (Bareiss) and re-checked',
            jacobian_file=jac_file, jacobian_payload_sha256=jac_sha, attempts=attempts),
        upper_bound=dict(value=ub['upper'], machine_certificate=ub, **manuscript_upper(n, info.get('_kind'))),
        certified_r=lower if certified else None,
        status='CERTIFIED_EQUAL' if certified else 'INTERVAL',
        interval=[lower, ub['upper']],
        exact_consistency_at_theta_star=dict(
            null_vectors_annihilate_all_rows=null_ok, exact_rank_at_theta_star=exact_rank_at_point,
            note='one-point consistency (rank at theta* equals the certified generic value), not an extra proof'),
        cross_checks=dict(
            features_exact_vs_float_ir_engine_max_abs=err_float,
            features_exact_vs_dense_source_statevector_max_abs=err_dense,
            features_exact_vs_qiskit_source_statevector_max_abs=err_qiskit,
            gradient_exact_vs_float_tangent_max_abs=err_grad,
            exact_d2_first_pairs=[str(v) for v in d2[:2]]),
        numerical_not_certified=num,
        timings_seconds=timings, peak_rss_mb_process=peak_rss_mb(),
        timing_note=('exact states/features are computed once per n for the union (weight<=2) readout '
                     'and cached; exact_states_total_cache is cumulative and is charged to the first '
                     'readout processed at that n'))
    return res


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------
def environment():
    import sympy
    env = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
               machine=platform.machine(), numpy=np.__version__, sympy=sympy.__version__)
    try:
        import qiskit
        env['qiskit'] = qiskit.__version__
    except Exception as e:  # pragma: no cover
        env['qiskit'] = 'unavailable: %r' % (e,)
    try:
        import scipy
        env['scipy'] = scipy.__version__
    except Exception:  # pragma: no cover
        pass
    env['threads'] = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'PYTHONHASHSEED')}
    return env


def provenance():
    files = []
    for p in sorted(CORE.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:
            rel = p.relative_to(CORE)
            orig = ORIGINAL_CORE / rel
            files.append(dict(path='core_copy/' + str(rel), sha256=sha256_file(p),
                              original='cross_family_resource_v1/core/' + str(rel),
                              original_sha256=sha256_file(orig) if orig.exists() else None))
    scope = json.loads((FROZEN / 'scope.json').read_text())
    frozen = [dict(id=s['id'], file_sha256=s['file_sha256']) for s in scope if s['module'] == MODULE]
    return dict(copied_core=files, copied_core_identical=all(f['sha256'] == f['original_sha256'] for f in files),
                code={p.name: sha256_file(p) for p in sorted(HERE.glob('*.py'))},
                frozen_kyriienko_inputs=frozen,
                manuscript_files={p: sha256_file(MANUSCRIPT / p) for p in
                                  ('family_extension.tex', 'fourier_gram.tex', 'dimension_theory.tex')})


def gate_ir_check(n, kind):
    info = make_info(n, kind)
    name = '%s_n%d_l1_%s' % (MODULE, n, kind)
    frozen_path = FROZEN / (name + '.json')
    scope = {s['id']: s for s in json.loads((FROZEN / 'scope.json').read_text())}
    mine = sha256_bytes(ir_text(info).encode())
    return info, dict(setting=name, generated_sha256=mine, frozen_file_sha256=scope[name]['file_sha256'],
                      frozen_file_actual_sha256=sha256_file(frozen_path),
                      dict_equal=json.loads(frozen_path.read_text()) == info,
                      sha_equal=mine == scope[name]['file_sha256'] == sha256_file(frozen_path))


def run(sizes, out, n_points=10, numeric=True, readouts=READOUTS):
    out.mkdir(parents=True, exist_ok=False)
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)
    t_start = time.perf_counter()
    save_json(out / 'environment.json', environment())
    save_json(out / 'provenance.json', provenance())
    euler = euler_identity_symbolic()
    save_json(out / 'euler_identity.json', euler)
    log('Euler identity (SymPy): %s' % euler['passed'])
    if not euler['passed']:
        raise SystemExit('Euler identity check failed')
    (out / 'ir').mkdir()
    gate = dict(ir=[], ranks=[], passed=None)
    results = []
    order = sorted(sizes)
    for n in order:
        log('== n=%d' % n)
        infos = {}
        for kind in readouts:
            if n in (2, 4):
                info, chk = gate_ir_check(n, kind)
                gate['ir'].append(chk)
                log('  IR gate %s: sha_equal=%s dict_equal=%s' % (chk['setting'], chk['sha_equal'], chk['dict_equal']))
            else:
                info = make_info(n, kind)
            (out / 'ir' / ('kyriienko_n%d_%s.json' % (n, kind))).write_text(ir_text(info))
            info = dict(info, _kind=kind)
            infos[kind] = info
        union = make_info(n, 'pauli_weight_le_2')
        cache = PointCache(union)
        for kind in readouts:
            name = 'kyriienko_n%d_%s' % (n, kind)
            res = certify_setting(infos[kind], cache, n_points=n_points, out_dir=out, name=name,
                                  numeric=numeric, log=log)
            res['source_route'] = source_ir(n)['meta']
            save_json(out / (name + '.json'), res)
            results.append(res)
            log('  %s: lower=%s upper=%s status=%s numeric=%s (%.1fs)' % (
                name, res['certified_lower_bound']['value'], res['upper_bound']['value'], res['status'],
                res['numerical_not_certified']['rank'] if res['numerical_not_certified'] else None,
                res['timings_seconds']['total']))
            if n in (2, 4):
                expected = GATE_PRIMARY.get((n, kind), GATE_SECONDARY.get((n, kind)))
                gate['ranks'].append(dict(setting=name, expected=expected, certified=res['certified_r'],
                                          primary=(n, kind) in GATE_PRIMARY,
                                          match=res['certified_r'] == expected))
        if n == 4 or (n > 4 and gate['passed'] is None):
            gate['passed'] = (bool(gate['ir']) and all(c['sha_equal'] and c['dict_equal'] for c in gate['ir'])
                              and all(r['match'] for r in gate['ranks'] if r['primary'])
                              and {(2, 'local_xyz'), (4, 'local_xyz'), (4, 'pauli_weight_le_2')} <=
                              {(int(r['setting'].split('_n')[1].split('_')[0]), r['setting'].split('_', 2)[2])
                               for r in gate['ranks']})
            gate['secondary_all_match'] = all(r['match'] for r in gate['ranks'] if not r['primary'])
            save_json(out / 'validation_gate.json', gate)
            log('VALIDATION GATE passed=%s secondary_all_match=%s' % (gate['passed'], gate['secondary_all_match']))
            if not gate['passed']:
                log('STOP: validation gate failed; larger sizes are not trusted and were not run.')
                break
    summary = dict(
        family=MODULE, sizes=order, validation_gate_passed=gate['passed'],
        table=[dict(n=r['n'], readout=r['readout'], K=r['K'], conjecture_3n_minus_2=r['conjecture_3n_minus_2'],
                    certified_lower=r['certified_lower_bound']['value'], upper=r['upper_bound']['value'],
                    certified_r=r['certified_r'], status=r['status'],
                    numerical_rank_not_certified=(r['numerical_not_certified'] or {}).get('rank'),
                    minor_digits=r['certified_lower_bound']['integer_minor_digits'],
                    seconds=r['timings_seconds']['total']) for r in results],
        wall_seconds=time.perf_counter() - t_start, peak_rss_mb_process=peak_rss_mb())
    save_json(out / 'summary.json', summary)
    (out / 'log.txt').write_text('\n'.join(log_lines) + '\n')
    save_json(out / 'manifest.json', dict(files=[dict(path=str(p.relative_to(out)), sha256=sha256_file(p))
                                                  for p in sorted(out.rglob('*')) if p.is_file()]))
    log('wall %.1fs, peak RSS %.0f MB' % (summary['wall_seconds'], summary['peak_rss_mb_process']))
    rep = verify_results(out)  # replay from the stored files (written after, outside the manifest)
    save_json(out / 'audit_replay.json', rep)
    print('audit replay passed=%s' % rep['passed'], flush=True)
    return summary


def verify_results(folder):
    """Audit replay from stored files only: manifest hashes, the stored integer Jacobian, and the
    minor determinant recomputed with SymPy DomainMatrix (independent of bareiss_det)."""
    folder = Path(folder)
    man = json.loads((folder / 'manifest.json').read_text())
    bad = [f['path'] for f in man['files'] if sha256_file(folder / f['path']) != f['sha256']]
    report = dict(manifest_mismatches=bad, settings=[])
    for p in sorted(folder.glob('kyriienko_n*_*.json')):
        if p.name.endswith('_jacobian.json.gz'):
            continue
        r = json.loads(p.read_text())
        lb = r['certified_lower_bound']
        raw = gzip.decompress((folder / lb['jacobian_file']).read_bytes())
        jac = json.loads(raw)
        rows = [[int(v) for v in row] for row in jac['integer_rows']]
        scales = [F(s) for s in jac['row_scales']]
        M = [[rows[i][k] for k in lb['minor_columns_theta']] for i in lb['minor_rows_pair_indices']]
        d = sympy_det(M) if M else 1
        scale = F(1)
        for i in lb['minor_rows_pair_indices']:
            scale *= scales[i]
        report['settings'].append(dict(
            setting=r['setting'], payload_sha256_ok=sha256_bytes(raw) == lb['jacobian_payload_sha256'],
            minor_order=len(M), stored_value=lb['value'],
            integer_det_matches=str(d) == lb['integer_minor_determinant'],
            rational_det_matches=str(scale * d) == lb['rational_minor_determinant'],
            nonzero=d != 0, certified_r=r['certified_r'], upper=r['upper_bound']['value']))
    report['passed'] = (not bad and all(s['payload_sha256_ok'] and s['integer_det_matches'] and
                                        s['rational_det_matches'] and (s['nonzero'] or s['minor_order'] == 0)
                                        and s['minor_order'] == s['stored_value'] for s in report['settings']))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sizes', type=int, nargs='+', default=[2, 4, 6, 8])
    ap.add_argument('--points', type=int, default=10)
    ap.add_argument('--no-numeric', action='store_true')
    ap.add_argument('--readouts', nargs='+', default=list(READOUTS))
    ap.add_argument('--output', type=Path)
    ap.add_argument('--verify', type=Path, help='audit an existing results folder and exit')
    a = ap.parse_args(argv)
    if a.verify:
        rep = verify_results(a.verify)
        print(json.dumps(rep, indent=2))
        return 0 if rep['passed'] else 1
    if any(n in (6, 8) for n in a.sizes) and not {2, 4} <= set(a.sizes):
        raise SystemExit('The validation gate (n = 2, 4) must run before n = 6, 8.')
    out = a.output or HERE / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    s = run(a.sizes, out, n_points=a.points, numeric=not a.no_numeric, readouts=a.readouts)
    return 0 if s['validation_gate_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
