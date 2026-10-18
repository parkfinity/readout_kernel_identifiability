"""Fourier-Gram exact rank certification for fixed-frequency projected kernels.

Operational core (theory drafted separately).  For a frozen compiler IR of
certified_dimension_extension_v1 (read only; the two reused engines are copied
unmodified into ./copied_from_extension_v1):

1. fixed-frequency check: no rotation-angle term multiplies a trainable symbol
   by a data angle (a term [c, t, x] with t >= 0 and x >= 0).
2. data-Fourier coefficients C(theta) of every measured feature,
       f_O(x; theta) = c_O(theta) + sum_a C_{O a}(theta) h_a(x),
   with h = {cos(w.psi), sin(w.psi) : w in Omega_+}, psi_b = phi_b(x)/M_b.
   * LaurentEngine: exact state propagation over Q(i)[u_b^{+-1}]
     (u_b = exp(i psi_b)) at a parameter point theta* whose half-angle phases
     exp(i theta_k / L_k) = (a_k + i b_k)/c_k are Gaussian rationals
     (Pythagorean triples).  mode='mod' reduces every rational modulo a prime
     p (all settings); mode='exact' keeps Python integers with a common
     denominator (selected settings).  Hadamards are propagated as the integer
     matrix sqrt(2) H and the factor 2^{-#H} is restored in the quadratic
     features.
   * NumericEngine: float statevector sampled on an aliasing-free DFT grid
     (diagnostics and numerical ranks only; never a certificate).
3. G(theta) = C^T W C and the Jacobian of theta -> upper(G).  If 1 and the h_a
   are linearly independent on the data domain, d^2 = Dh^T G Dh determines G
   and conversely (injectivity lemma), so rank J_{d^2}(theta) = rank J_G(theta)
   at every theta.
4. Certificates.
   lower: a nonzero r x r minor of J_G(theta*), exact (rational) or modulo a
          prime (an integer minor that is nonzero mod p is nonzero).
   upper: (U1) null fields of J_G valid for all theta, from two-copy
          invariance of Omega = sum_O w_O O (x) O: iterative deletion of
          rotations whose output-transported generator P satisfies
          [Omega, P(x)I + I(x)P] = 0 (or the commutant condition for data
          angles), plus "virtual insertion" fields of a single-qubit Pauli Q
          after a three-rotation Euler block (Cramer field, identity verified
          by exact polynomial reduction in sympy).  r <= K - rank(family(theta*)).
          (U2) symbolic two-copy Heisenberg expansion of
          K(x,x') = Tr[(rho(x) (x) rho(x')) Omega] with theta kept symbolic:
          exact per-symbol inactivity and the bound r <= dim span of the
          non-constant parts of the entries of G.  Budget limited.
Numerical identities from random all-zero tests are never accepted.
"""
from collections import Counter
from fractions import Fraction as F
import itertools
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'copied_from_extension_v1'))
import exact_engine as ex  # noqa: E402  (copied, unmodified)

BASE_COORD = {'a0': 0, 'a1': 1, 'acos(a0)': 0, 'acos(a1)': 1}
BASE_IMAGE = {'a0': '(-1, 1)', 'a1': '(-1, 1)', 'acos(a0)': '(0, pi)', 'acos(a1)': '(0, pi)'}
PRIMES = (134217689, 134217649)  # two largest primes below 2^27


def lcm(a, b):
    return a * b // math.gcd(a, b)


def frac_str(x):
    return str(F(x))


# ----------------------------------------------------------------------------
# IR parsing and fixed-frequency analysis
# ----------------------------------------------------------------------------
class Setting:
    def __init__(self, info, name=''):
        ex.validate(info)
        self.info, self.name = info, name
        self.n, self.K = info['n'], info['count']
        self.N = 2 ** self.n
        self.obs = list(info['observables'])
        self.w = [F(str(v)) for v in info['weights']]
        bases, angles = ex.canonical_angles(info)
        self.bases, self.B = bases, len(bases)
        self.gates, self.product_terms, self.constant_terms = [], [], []
        for idx, (g, ang) in enumerate(zip(info['program'], angles)):
            if g['op'] != 'R':
                self.gates.append(dict(op=g['op'], wires=tuple(g['wires']), src=idx))
                continue
            th, da = {}, {}
            for t, slot, c in ang:
                if t >= 0 and slot >= 0:
                    self.product_terms.append(dict(gate=idx, symbol=t, base=bases[slot], coefficient=str(c)))
                elif t >= 0:
                    th[t] = th.get(t, F(0)) + c
                elif slot >= 0:
                    da[slot] = da.get(slot, F(0)) + c
                else:
                    self.constant_terms.append(dict(gate=idx, coefficient=str(c)))
            self.gates.append(dict(op='R', pauli=g['pauli'], src=idx,
                                   th={k: v for k, v in th.items() if v},
                                   da={b: v for b, v in da.items() if v}))
        self.fixed_frequency = not self.product_terms
        self.exact_supported = self.fixed_frequency and not self.constant_terms
        self.nH = sum(g['op'] == 'H' for g in self.gates)
        rot = [g for g in self.gates if g['op'] == 'R']
        # data units: psi_b = phi_b / M_b, half-angle exponents s_{g,b} are integers
        self.M = [1] * self.B
        for g in rot:
            for b, k in g['da'].items():
                self.M[b] = lcm(self.M[b], (k / 2).denominator)
        for g in rot:
            g['s'] = tuple(int(g['da'].get(b, 0) / 2 * self.M[b]) for b in range(self.B))
        self.gstep, self.S, self.E = [], [], []
        for b in range(self.B):
            vals = [abs(g['s'][b]) for g in rot if g['s'][b]]
            gs = 0
            for v in vals:
                gs = math.gcd(gs, v)
            gs = gs or 1
            self.gstep.append(gs)
            self.S.append(sum(vals))
            self.E.append(2 * sum(vals) // gs + 1)
        # parameter units: exp(i theta_k / L_k) is the Gaussian-rational phase
        self.L = [1] * self.K
        for g in rot:
            for k, c in g['th'].items():
                self.L[k] = lcm(self.L[k], (c / 2).denominator)
        self.Lam = 1
        for g in rot:
            g['m'] = {k: int(c / 2 * self.L[k]) for k, c in g['th'].items()}
            for c in g['th'].values():
                self.Lam = lcm(self.Lam, (c / 2).denominator)
        self.symbol_gates = {k: [i for i, g in enumerate(self.gates) if g['op'] == 'R' and k in g['th']]
                             for k in range(self.K)}
        # Omega_+ in exponent units on the omega grid of shape (2E_b-1)
        self.wshape = tuple(2 * e - 1 for e in self.E)
        self.omega_plus = []
        for jj in np.ndindex(*self.wshape):
            v = tuple((j - (e - 1)) * gs for j, e, gs in zip(jj, self.E, self.gstep))
            nz = [x for x in v if x]
            if nz and nz[0] > 0:
                neg = tuple((e - 1) - (j - (e - 1)) for j, e in zip(jj, self.E))
                self.omega_plus.append((int(np.ravel_multi_index(jj, self.wshape)),
                                        int(np.ravel_multi_index(neg, self.wshape)), v))
        self.basis = [(kind, v) for _, _, v in self.omega_plus for kind in ('cos', 'sin')]
        self._ptab = {}

    # -- descriptive records ------------------------------------------------
    def describe(self):
        coords = [BASE_COORD[b] for b in self.bases]
        return dict(
            fixed_frequency=self.fixed_frequency, product_terms=self.product_terms,
            constant_angle_terms=self.constant_terms, exact_supported=self.exact_supported,
            data_bases=self.bases, base_units_M=self.M, half_angle_step=self.gstep,
            max_half_angle_exponent=self.S,
            dft_grid_points_per_base=[2 * e - 1 for e in self.E],
            max_feature_frequency_in_psi_units=[2 * s for s in self.S],
            basis_full_size=len(self.basis),
            basis_description=('h_a(x) in {cos(w.psi), sin(w.psi)}, psi_b = phi_b(x)/M_b, '
                               'w in Omega_+ (first nonzero component > 0), constant mode removed'),
            independence=dict(
                bases_on_distinct_coordinates=len(set(coords)) == len(coords),
                images=[BASE_IMAGE[b] for b in self.bases],
                argument=('phi maps the open domain (-1,1)^2 onto an open box; 1 and the distinct '
                          'characters exp(i w.psi) are linearly independent on any open set '
                          '(real-analytic trigonometric polynomial vanishing on an open set vanishes '
                          'identically), hence so are 1, cos(w.psi), sin(w.psi), w in Omega_+.')),
            grid_sufficiency=('each rotation exp(-i alpha P/2) multiplies amplitudes by '
                              'u^{-s} or u^{+s} (data part of alpha/2), so amplitudes have '
                              'exponents in [-S_b, S_b] (step g_b) and features have '
                              'frequencies |w_b| <= 2 S_b; an equispaced grid of 2(2S_b/g_b)+1 '
                              'points in chi_b = g_b psi_b is aliasing free.'),
            H_gates=self.nH)

    def pauli_table(self, word):
        if word not in self._ptab:
            j = np.arange(self.N)
            tgt = j.copy()
            code = np.zeros(self.N, dtype=np.int64)
            for q, a in enumerate(word):
                bit = (j >> q) & 1
                if a in 'XY':
                    tgt ^= 1 << q
                if a == 'Y':
                    code += np.where(bit == 1, 3, 1)
                elif a == 'Z':
                    code += np.where(bit == 1, 2, 0)
            self._ptab[word] = (tgt, code % 4)
        return self._ptab[word]

    def base_angles(self, x):
        """phi_b(x) as floats for x in (-1,1)^2."""
        return [math.acos(x[BASE_COORD[b]]) if b.startswith('acos') else x[BASE_COORD[b]] for b in self.bases]

    def h(self, x):
        psi = [p / m for p, m in zip(self.base_angles(x), self.M)]
        out = []
        for kind, v in self.basis:
            a = sum(vb * pb for vb, pb in zip(v, psi))
            out.append(math.cos(a) if kind == 'cos' else math.sin(a))
        return np.array(out)


# ----------------------------------------------------------------------------
# Exact parameter points with Gaussian-rational half-angle phases
# ----------------------------------------------------------------------------
def primitive_triples(cmax=400):
    out = []
    for m in range(2, int(math.isqrt(cmax)) + 2):
        for k in range(1, m):
            if (m - k) % 2 == 1 and math.gcd(m, k) == 1:
                a, b, c = m * m - k * k, 2 * m * k, m * m + k * k
                if c <= cmax:
                    out.append((a, b, c))
    return sorted(out, key=lambda t: (t[2], t[0]))


def pythagorean_point(st, seed):
    rng = random.Random(seed)
    pick = rng.sample(primitive_triples(), st.K)
    zeta = []
    for a, b, c in pick:
        if rng.random() < 0.5:
            a, b = b, a
        zeta.append((a * rng.choice((1, -1)), b * rng.choice((1, -1)), c))
    theta = [st.L[k] * math.atan2(b, a) for k, (a, b, c) in enumerate(zeta)]
    spec = ['theta_%d = %d*atan2(%d, %d)  [exp(i theta/%d) = (%d + %d i)/%d]'
            % (k, st.L[k], b, a, st.L[k], a, b, c) for k, (a, b, c) in enumerate(zeta)]
    return dict(seed=seed, zeta=zeta, L=list(st.L), theta_float=theta, specification=spec)


def gpow(a, b, m):
    """(a + i b)^m for m >= 0, (a - i b)^{|m|} for m < 0, as integer pair."""
    if m < 0:
        b, m = -b, -m
    r, i = 1, 0
    for _ in range(m):
        r, i = r * a - i * b, r * b + i * a
    return r, i


def half_angle_cs(point, k, m):
    """Exact (cos, sin) of m*theta_k/L_k as Fractions."""
    a, b, c = point['zeta'][k]
    r, i = gpow(a, b, m)
    return F(r, c ** abs(m)), F(i, c ** abs(m))


# ----------------------------------------------------------------------------
# Laurent-polynomial state propagation (mod p or exact)
# ----------------------------------------------------------------------------
class LaurentEngine:
    def __init__(self, st, point, mode='mod', p=PRIMES[0]):
        if not st.exact_supported:
            raise ValueError('Exact engine requires a fixed-frequency IR without constant angles')
        self.st, self.point, self.mode, self.p = st, point, mode, p
        self.dtype = np.int64 if mode == 'mod' else object
        self.zph = {}
        for i, g in enumerate(st.gates):
            if g['op'] != 'R':
                continue
            zr, zi, cz = 1, 0, 1
            for k, m in g['m'].items():   # theta part of exp(-i alpha/2) = prod zeta_k^{-m}
                a, b, c = point['zeta'][k]
                r, s = gpow(a, b, -m)
                zr, zi = zr * r - zi * s, zr * s + zi * r
                cz *= c ** abs(m)
            self.zph[i] = (zr, zi, cz)

    def red(self, x):
        return x % self.p if self.mode == 'mod' else x

    def sc(self, v):
        return v % self.p if self.mode == 'mod' else v

    def zeros(self, shape):
        return np.zeros(shape, dtype=self.dtype)

    def pauli(self, re, im, word):
        tgt, code = self.st.pauli_table(word)
        nr = np.where(code == 0, re, np.where(code == 1, -im, np.where(code == 2, -re, im)))
        ni = np.where(code == 0, im, np.where(code == 1, re, np.where(code == 2, -im, -re)))
        orr, oi = self.zeros(re.shape), self.zeros(re.shape)
        orr[..., tgt] = nr
        oi[..., tgt] = ni
        return self.red(orr), self.red(oi)

    def shift(self, arr, d, lead):
        out = self.zeros(arr.shape)
        src, dst = [slice(None)] * lead, [slice(None)] * lead
        for b, s in enumerate(d):
            e, k = self.st.E[b], s // self.st.gstep[b]
            if k >= 0:
                src.append(slice(0, e - k)); dst.append(slice(k, e))
            else:
                src.append(slice(-k, e)); dst.append(slice(0, e + k))
        out[tuple(dst)] = arr[tuple(src)]
        return out

    def rot(self, re, im, i, g, lead):
        pr, pi = self.pauli(re, im, g['pauli'])
        ar, ai = re + pr, im + pi      # 2 Pi_+ psi
        br, bi = re - pr, im - pi      # 2 Pi_- psi
        zr, zi, cz = self.zph[i]
        zr, zi = self.sc(zr), self.sc(zi)
        xr, xi = self.red(ar * zr - ai * zi), self.red(ar * zi + ai * zr)   # * z
        yr, yi = self.red(br * zr + bi * zi), self.red(bi * zr - br * zi)   # * conj(z)
        s = g['s']
        neg = tuple(-v for v in s)
        nr = self.red(self.shift(xr, neg, lead) + self.shift(yr, s, lead))
        ni = self.red(self.shift(xi, neg, lead) + self.shift(yi, s, lead))
        return nr, ni, 2 * cz

    def clifford(self, arr, g):
        n = self.st.N
        if g['op'] == 'CX':
            c, t = g['wires']
            j = np.arange(n)
            src = np.where(j & (1 << c), j ^ (1 << t), j)
            return arr[..., src]
        bit = 1 << g['wires'][0]
        j = np.arange(n)
        lo = (j & bit) == 0
        out = self.zeros(arr.shape)
        out[..., j[lo]] = arr[..., j[lo]] + arr[..., j[lo] ^ bit]
        out[..., j[~lo]] = arr[..., j[~lo] ^ bit] - arr[..., j[~lo]]
        return self.red(out)

    def run(self):
        st = self.st
        shape = tuple(st.E) + (st.N,)
        re, im = self.zeros(shape), self.zeros(shape)
        centre = tuple((e - 1) // 2 for e in st.E)
        re[centre + (0,)] = 1
        tr, ti = self.zeros((st.K,) + shape), self.zeros((st.K,) + shape)
        den = 1
        for i, g in enumerate(st.gates):
            if g['op'] != 'R':
                re, im = self.clifford(re, g), self.clifford(im, g)
                tr, ti = self.clifford(tr, g), self.clifford(ti, g)
                continue
            re, im, f = self.rot(re, im, i, g, 0)
            tr, ti, _ = self.rot(tr, ti, i, g, 1)
            den *= f
            if g['th']:
                pr, pi = self.pauli(re, im, g['pauli'])   # -i P psi_new = (pi, -pr)
                for k, c in g['th'].items():
                    h = c / 2
                    fac = self.sc(h.numerator * (st.Lam // h.denominator))
                    tr[k] = self.red(tr[k] + pi * fac)
                    ti[k] = self.red(ti[k] - pr * fac)
        self.psi, self.tan, self.den, self.den_t = (re, im), (tr, ti), den, den * st.Lam
        return self

    # -- modular feature coefficients on the full omega grid ----------------
    def coefficients_mod(self):
        st, p = self.st, self.p
        etot = int(np.prod(st.E))
        idx = np.array(list(np.ndindex(*st.E)))
        dif = idx[None, :, :] - idx[:, None, :] + (np.array(st.E) - 1)[None, None, :]
        binidx = np.ravel_multi_index(tuple(dif[..., b] for b in range(st.B)), st.wshape).ravel()
        nbin = int(np.prod(st.wshape))
        pr = self.psi[0].reshape(etot, st.N)
        pim = self.psi[1].reshape(etot, st.N)
        tr = self.tan[0].reshape(st.K, etot, st.N)
        ti = self.tan[1].reshape(st.K, etot, st.N)
        sc_f = pow(self.den * self.den * 2 ** st.nH, -1, p)
        sc_t = pow(self.den * self.den_t * 2 ** st.nH, -1, p)

        def binned(S):
            v = np.bincount(binidx, weights=(S % p).reshape(-1).astype(np.float64), minlength=nbin)
            return np.rint(v).astype(np.int64) % p

        def binned_k(S):
            return np.stack([binned(S[k]) for k in range(S.shape[0])])

        O = len(st.obs)
        Fr = np.zeros((O, nbin), np.int64); Fi = np.zeros((O, nbin), np.int64)
        Tr = np.zeros((O, st.K, nbin), np.int64); Ti = np.zeros((O, st.K, nbin), np.int64)
        for o, word in enumerate(st.obs):
            qr, qi = self.pauli(pr, pim, word)
            Fr[o] = binned(pr @ qr.T + pim @ qi.T)
            Fi[o] = binned(pr @ qi.T - pim @ qr.T)
            ur, ui = self.pauli(tr, ti, word)
            Tr[o] = binned_k(np.einsum('en,kfn->kef', pr, ur) + np.einsum('en,kfn->kef', pim, ui))
            Ti[o] = binned_k(np.einsum('en,kfn->kef', pr, ui) - np.einsum('en,kfn->kef', pim, ur))
        Fr, Fi = Fr * sc_f % p, Fi * sc_f % p
        Tr, Ti = Tr * sc_t % p, Ti * sc_t % p
        pos = np.array([a for a, _, _ in st.omega_plus], dtype=np.int64)
        neg = np.array([b for _, b, _ in st.omega_plus], dtype=np.int64)
        real_ok = bool(np.all(Fr[:, neg] == Fr[:, pos]) and np.all((Fi[:, neg] + Fi[:, pos]) % p == 0))
        # d f / d theta_k has coefficient T[w] + conj(T[-w])
        Dr, Di = (Tr[:, :, pos] + Tr[:, :, neg]) % p, (Ti[:, :, pos] - Ti[:, :, neg]) % p
        A = 2 * len(pos)
        C = np.zeros((O, A), np.int64); dC = np.zeros((st.K, O, A), np.int64)
        C[:, 0::2], C[:, 1::2] = 2 * Fr[:, pos] % p, (-2 * Fi[:, pos]) % p
        dC[:, :, 0::2] = np.transpose(2 * Dr % p, (1, 0, 2))
        dC[:, :, 1::2] = np.transpose((-2 * Di) % p, (1, 0, 2))
        return C, dC, real_ok

    # -- exact feature coefficients on the reachable support ----------------
    def coefficients_exact(self):
        st = self.st
        pr = self.psi[0].reshape(-1, st.N); pim = self.psi[1].reshape(-1, st.N)
        tr = self.tan[0].reshape(st.K, -1, st.N); ti = self.tan[1].reshape(st.K, -1, st.N)
        nz = lambda a: np.array([any(v != 0 for v in row) for row in a])
        sup = nz(pr) | nz(pim)
        for k in range(st.K):
            sup |= nz(tr[k]) | nz(ti[k])
        sup = np.where(sup)[0]
        expo = [tuple((j - (e - 1) // 2) * gs for j, e, gs in zip(np.unravel_index(s, st.E), st.E, st.gstep))
                for s in sup]
        omega_index = {v: n for n, (_, _, v) in enumerate(st.omega_plus)}
        diffs = {}
        for a, ea in enumerate(expo):
            for b, eb in enumerate(expo):
                diffs.setdefault(tuple(y - x for x, y in zip(ea, eb)), []).append((sup[a], sup[b]))
        den_f = self.den * self.den * 2 ** st.nH
        den_t = self.den * self.den_t * 2 ** st.nH
        O = len(st.obs)
        res = {}

        def corr(ar, ai, br, bi, plist):
            r = i = 0
            for a, b in plist:
                r += int(np.dot(ar[a], br[b]) + np.dot(ai[a], bi[b]))
                i += int(np.dot(ar[a], bi[b]) - np.dot(ai[a], br[b]))
            return r, i

        for o, word in enumerate(st.obs):
            qr, qi = self.pauli(pr, pim, word)
            ur, ui = self.pauli(tr, ti, word)
            for w, n in omega_index.items():
                negw = tuple(-v for v in w)
                pl, ml = diffs.get(w, []), diffs.get(negw, [])
                if not pl and not ml:
                    continue
                fr, fi = corr(pr, pim, qr, qi, pl)
                cos_c, sin_c = F(2 * fr, den_f), F(-2 * fi, den_f)
                dcos, dsin = [], []
                for k in range(st.K):
                    a_r, a_i = corr(pr, pim, ur[k], ui[k], pl)
                    b_r, b_i = corr(pr, pim, ur[k], ui[k], ml)
                    dcos.append(F(2 * (a_r + b_r), den_t))
                    dsin.append(F(-2 * (a_i - b_i), den_t))
                res[(o, n)] = (cos_c, sin_c, dcos, dsin)
        cols = sorted({n for _, n in res})
        A = 2 * len(cols)
        C = [[F(0)] * A for _ in range(O)]
        dC = [[[F(0)] * A for _ in range(O)] for _ in range(st.K)]
        for (o, n), (cc, ss, dc, ds) in res.items():
            j = cols.index(n)
            C[o][2 * j], C[o][2 * j + 1] = cc, ss
            for k in range(st.K):
                dC[k][o][2 * j], dC[k][o][2 * j + 1] = dc[k], ds[k]
        labels = [(kind, st.omega_plus[n][2]) for n in cols for kind in ('cos', 'sin')]
        return C, dC, labels


# ----------------------------------------------------------------------------
# Modular / exact linear algebra
# ----------------------------------------------------------------------------
def gram_jacobian_mod(C, dC, weights, p):
    """J[(a<=b), k] = d_k G_ab mod p, restricted to support columns."""
    w = np.array([int(F(x).numerator * pow(F(x).denominator, -1, p)) % p for x in weights], np.int64)
    support = np.where((C != 0).any(0) | (dC != 0).any((0, 1)))[0]
    Cs, dCs = C[:, support], dC[:, :, support]
    Cw = Cs * w[:, None] % p
    A = len(support)
    iu = np.triu_indices(A)
    J = np.zeros((len(iu[0]), dC.shape[0]), np.int64)
    for k in range(dC.shape[0]):
        Gk = (dCs[k].T @ Cw % p + Cw.T @ dCs[k] % p) % p
        J[:, k] = Gk[iu]
    return J, support, iu


def rank_mod(J, p):
    M = J.copy() % p
    R, K = M.shape
    used = np.zeros(R, bool)
    piv_rows, piv_cols = [], []
    for c in range(K):
        cand = np.where((M[:, c] != 0) & ~used)[0]
        if len(cand) == 0:
            continue
        r = int(cand[0])
        used[r] = True
        piv_rows.append(r); piv_cols.append(c)
        inv = pow(int(M[r, c]), -1, p)
        M[r] = M[r] * inv % p
        f = M[:, c].copy(); f[r] = 0
        nzr = np.where(f != 0)[0]
        if len(nzr):
            M[nzr] = (M[nzr] - (f[nzr, None] * M[r][None, :]) % p) % p
    return len(piv_rows), piv_rows, piv_cols


def det_mod(rows, p):
    M = [[int(v) % p for v in r] for r in rows]
    n, d = len(M), 1
    for c in range(n):
        r = next((i for i in range(c, n) if M[i][c]), None)
        if r is None:
            return 0
        if r != c:
            M[c], M[r] = M[r], M[c]; d = -d
        d = d * M[c][c] % p
        inv = pow(M[c][c], -1, p)
        for i in range(c + 1, n):
            f = M[i][c] * inv % p
            if f:
                M[i] = [(x - f * y) % p for x, y in zip(M[i], M[c])]
    return d % p


def rank_exact(rows, ncols):
    """Exact rank over Q of a list of Fraction rows; returns rank, pivot rows, pivot cols."""
    M = [list(r) for r in rows]
    used = [False] * len(M)
    pr, pc = [], []
    for c in range(ncols):
        r = next((i for i in range(len(M)) if not used[i] and M[i][c] != 0), None)
        if r is None:
            continue
        used[r] = True
        pr.append(r); pc.append(c)
        pv = M[r][c]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c] / pv
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
    return len(pr), pr, pc


def det_exact(rows):
    M = [list(r) for r in rows]
    n, d = len(M), F(1)
    for c in range(n):
        r = next((i for i in range(c, n) if M[i][c] != 0), None)
        if r is None:
            return F(0)
        if r != c:
            M[c], M[r] = M[r], M[c]; d = -d
        d *= M[c][c]
        for i in range(c + 1, n):
            f = M[i][c] / M[c][c]
            if f:
                M[i] = [x - f * y for x, y in zip(M[i], M[c])]
    return d


def frac_mod(x, p):
    x = F(x)
    return x.numerator * pow(x.denominator, -1, p) % p


# ----------------------------------------------------------------------------
# Numerical statevector + DFT engine (diagnostics only)
# ----------------------------------------------------------------------------
class NumericEngine:
    def __init__(self, st):
        self.st = st

    def _pauli(self, s, word):
        tgt, code = self.st.pauli_table(word)
        ph = np.array([1, 1j, -1, -1j])[code]
        out = np.zeros_like(s)
        out[..., tgt] = s * ph
        return out

    def simulate(self, theta, phi, tangents=True):
        st = self.st
        P = phi.shape[0]
        s = np.zeros((P, st.N), complex); s[:, 0] = 1
        t = np.zeros((st.K, P, st.N), complex) if tangents else None
        for g in st.gates:
            if g['op'] == 'CX':
                c, tt = g['wires']
                j = np.arange(st.N)
                src = np.where(j & (1 << c), j ^ (1 << tt), j)
                s = s[..., src]
                if tangents:
                    t = t[..., src]
                continue
            if g['op'] == 'H':
                bit = 1 << g['wires'][0]
                j = np.arange(st.N); lo = (j & bit) == 0

                def H(v):
                    out = np.empty_like(v)
                    out[..., j[lo]] = (v[..., j[lo]] + v[..., j[lo] ^ bit]) / np.sqrt(2)
                    out[..., j[~lo]] = (v[..., j[~lo] ^ bit] - v[..., j[~lo]]) / np.sqrt(2)
                    return out
                s = H(s)
                if tangents:
                    t = H(t)
                continue
            ang = np.zeros(P)
            for k, c in g['th'].items():
                ang = ang + float(c) * theta[k]
            for b, c in g['da'].items():
                ang = ang + float(c) * phi[:, b]
            co, si = np.cos(ang / 2)[:, None], np.sin(ang / 2)[:, None]
            s_new = co * s - 1j * si * self._pauli(s, g['pauli'])
            if tangents:
                t = co * t - 1j * si * self._pauli(t, g['pauli'])
                ps_new = self._pauli(s_new, g['pauli'])
                for k, c in g['th'].items():
                    t[k] += -0.5j * float(c) * ps_new
            s = s_new
        f, df = [], []
        for o in st.obs:
            po = self._pauli(s, o)
            f.append(np.real(np.sum(s.conj() * po, -1)))
            if tangents:
                df.append(2 * np.real(np.sum(t.conj() * po[None], -1)))
        return np.array(f), (np.array(df) if tangents else None)

    def coefficients(self, theta):
        st = self.st
        Ns = st.wshape
        axes = [2 * np.pi * np.arange(n) / n for n in Ns]
        grid = np.meshgrid(*axes, indexing='ij')
        chi = np.stack([g.ravel() for g in grid], 1)
        phi = chi * np.array([m / gs for m, gs in zip(st.M, st.gstep)])[None, :]
        f, df = self.simulate(theta, phi)
        O, P = f.shape
        fh = np.fft.fftn(f.reshape((O,) + Ns), axes=tuple(range(1, st.B + 1))) / P
        dfh = np.fft.fftn(df.reshape((O, st.K) + Ns), axes=tuple(range(2, st.B + 2))) / P
        cols = [tuple((v // gs) % n for v, gs, n in zip(w, st.gstep, Ns)) for _, _, w in st.omega_plus]
        A = 2 * len(cols)
        C = np.zeros((O, A)); dC = np.zeros((st.K, O, A))
        for j, c in enumerate(cols):
            C[:, 2 * j] = 2 * fh[(slice(None),) + c].real
            C[:, 2 * j + 1] = -2 * fh[(slice(None),) + c].imag
            dC[:, :, 2 * j] = 2 * dfh[(slice(None), slice(None)) + c].real.T
            dC[:, :, 2 * j + 1] = -2 * dfh[(slice(None), slice(None)) + c].imag.T
        return C, dC

    def jacobian(self, theta, support=None):
        C, dC = self.coefficients(theta)
        if support is None:
            scale = max(np.abs(C).max(), np.abs(dC).max(), 1e-300)
            support = np.where((np.abs(C) > 1e-13 * scale).any(0) | (np.abs(dC) > 1e-13 * scale).any((0, 1)))[0]
        W = np.array([float(x) for x in self.st.w])
        Cs, dCs = C[:, support] * W[:, None] ** 0.5, dC[:, :, support] * W[None, :, None] ** 0.5
        iu = np.triu_indices(len(support))
        J = np.stack([(dCs[k].T @ Cs + Cs.T @ dCs[k])[iu] for k in range(self.st.K)], 1)
        return J, C, dC, support

    def distance_direct(self, theta, x, xp):
        st = self.st
        phi = np.array([st.base_angles(x), st.base_angles(xp)])
        f, _ = self.simulate(theta, phi, tangents=False)
        W = np.array([float(v) for v in st.w])
        return float(np.sum(W * (f[:, 0] - f[:, 1]) ** 2))

    def distance_gram(self, C, x, xp):
        dh = self.st.h(x) - self.st.h(xp)
        W = np.array([float(v) for v in self.st.w])
        G = C.T @ (W[:, None] * C)
        return float(dh @ G @ dh)


def numerical_rank(st, seeds=(11, 12, 13, 14), tol=1e-9):
    """NOT a certificate: SVD of the float Gram Jacobian at random theta."""
    eng = NumericEngine(st)
    rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        theta = rng.uniform(-np.pi, np.pi, st.K)
        J, _, _, support = eng.jacobian(theta)
        sv = np.linalg.svd(J, compute_uv=False) if J.size else np.zeros(0)
        top = float(sv[0]) if len(sv) and sv[0] > 0 else 0.0
        r = int(np.sum(sv > tol * top)) if top > 1e-12 else 0
        rows.append(dict(seed=s, rank=r, sigma_max=top,
                         sigma_r_rel=float(sv[r - 1] / top) if r else None,
                         sigma_r1_rel=float(sv[r] / top) if top and r < len(sv) else None,
                         support_columns=int(len(support))))
    return dict(certified=False, tolerance_relative=tol, trials=rows,
                rank=max(r['rank'] for r in rows), min_rank=min(r['rank'] for r in rows))


# ----------------------------------------------------------------------------
# Two-copy invariance certificates (upper bounds)
# ----------------------------------------------------------------------------
def omega_terms(st):
    return {o + o: (F(w), F(0)) for o, w in zip(st.obs, st.w)}


def commutator_zero(Om, gens):
    """[Omega, sum(gens)] == 0 exactly (Pauli algebra with exact phases)."""
    acc = {}
    for W, c in Om.items():
        for G in gens:
            if not ex.commutes(W, G):
                z, word = ex.product(W, G)
                ex.push(acc, word, ex.scale(ex.mul(c, z), F(2)))
    return not acc


def joint_invariant(st, Om, P):
    I = 'I' * st.n
    return commutator_zero(Om, [P + I, I + P])


def commutant_invariant(st, Om, P):
    I = 'I' * st.n
    return commutator_zero(Om, [P + I]) and commutator_zero(Om, [I + P])


def transport(word, later):
    sign = 1
    for g in later:
        if g['op'] in ('H', 'CX'):
            s, word = ex.conjugate(word, dict(op=g['op'], wires=list(g['wires'])))
            sign *= s
        elif not ex.commutes(word, g['pauli']):
            return None
    return sign, word


def certified_reduction(st):
    """Iterative deletion; each step keeps K(x,x';theta) unchanged for all inputs/parameters."""
    Om = omega_terms(st)
    work = [dict(g, th=dict(g['th']), da=dict(g['da'])) if g['op'] == 'R' else dict(g) for g in st.gates]
    log = []
    changed = True
    while changed:
        changed = False
        for i in range(len(work) - 1, -1, -1):
            g = work[i]
            if g['op'] != 'R':
                continue
            tr = transport(g['pauli'], work[i + 1:])
            if tr is None:
                continue
            sign, P = tr
            tau = (st.K + 2 * st.B, st.K + 2 * st.B + 1)
            if commutant_invariant(st, Om, P):
                snap = [dict(h) for h in work[i:]]
                snap[0] = dict(g, fresh=tau, fresh_keep_data=False)
                log.append(dict(_suffix=snap, source_gate=g['src'], rule='commutant: [Omega, P(x)I] = [Omega, I(x)P] = 0',
                                removed_symbols={str(k): str(v) for k, v in g['th'].items()},
                                removed_data={st.bases[b]: str(v) for b, v in g['da'].items()},
                                output_generator=P, sign=sign))
                del work[i]
                changed = True
                break
            if g['th'] and joint_invariant(st, Om, P):
                snap = [dict(h) for h in work[i:]]
                snap[0] = dict(g, fresh=(tau[0], tau[0]), fresh_keep_data=True)
                log.append(dict(_suffix=snap, source_gate=g['src'], rule='two-copy joint: [Omega, P(x)I + I(x)P] = 0',
                                removed_symbols={str(k): str(v) for k, v in g['th'].items()},
                                removed_data={}, output_generator=P, sign=sign))
                g['th'] = {}
                if not g['da']:
                    del work[i]
                changed = True
                break
    present = {k for g in work if g['op'] == 'R' for k in g['th']}
    return work, sorted(set(range(st.K)) - present), log


_EULER_CACHE = {}


def euler_field(paulis, Q):
    """Cramer field w (alpha coordinates) with sum_j w_j d_j B = (-i/2) det(M) Q B.

    B = R_{P3}(a3) R_{P2}(a2) R_{P1}(a1).  Variables Cj = cos(aj/2), Sj = sin(aj/2),
    ii = sqrt(-1).  The identity is verified by exact reduction modulo the Groebner
    basis {Cj^2 + Sj^2 - 1, ii^2 + 1}.  Returns None when det(M) reduces to 0.
    """
    key = (tuple(paulis), Q)
    if key in _EULER_CACHE:
        return _EULER_CACHE[key]
    C = sp.symbols('C0:3'); S = sp.symbols('S0:3'); ii = sp.Symbol('ii')
    sig = {'X': sp.Matrix([[0, 1], [1, 0]]), 'Y': sp.Matrix([[0, -ii], [ii, 0]]),
           'Z': sp.Matrix([[1, 0], [0, -1]])}
    I2 = sp.eye(2)
    ideal = [C[j] ** 2 + S[j] ** 2 - 1 for j in range(3)] + [ii ** 2 + 1]
    gens = [ii] + list(S) + list(C)

    def red(e):
        e = sp.expand(e)
        if e == 0:
            return sp.Integer(0)
        return sp.reduced(e, ideal, *gens, order='lex')[1]

    def dag(A):
        return A.T.subs(ii, -ii)

    R = [C[j] * I2 - ii * S[j] * sig[paulis[j]] for j in range(3)]
    B = R[2] * R[1] * R[0]
    after = [R[2] * R[1], R[2], I2]
    before = [I2, R[0], R[1] * R[0]]
    # d R_j / d a_j = (-ii/2) sigma_j R_j
    dB = [after[j] * (-ii / 2 * sig[paulis[j]]) * R[j] * before[j] for j in range(3)]
    q = []
    for j in range(3):
        A = after[j] * sig[paulis[j]] * dag(after[j])
        q.append([red((A * sig[a]).trace() / 2) for a in 'XYZ'])
    M = sp.Matrix(3, 3, lambda r, c: q[c][r])
    e = sp.Matrix([1 if a == Q else 0 for a in 'XYZ'])
    w = [red(v) for v in (M.adjugate() * e)]
    det = red(M.det())
    if det == 0:
        _EULER_CACHE[key] = None
        return None
    lhs = sum((w[j] * dB[j] for j in range(3)), sp.zeros(2, 2)) - (-ii / 2) * det * sig[Q] * B
    ok = all(red(v) == 0 for v in lhs)
    if not ok:
        raise AssertionError('Euler field identity failed for %s %s' % (paulis, Q))
    out = dict(w=w, det=det, symbols=(C, S), verified=True,
               w_text=[str(v) for v in w], det_text=str(det))
    _EULER_CACHE[key] = out
    return out


def virtual_fields(st, work):
    Om = omega_terms(st)
    count = Counter(k for g in work if g['op'] == 'R' for k in g['th'])
    out = []
    for q in range(st.n):
        touching = [i for i, g in enumerate(work)
                    if (g['op'] == 'R' and g['pauli'][q] != 'I') or (g['op'] != 'R' and q in g['wires'])]
        runs, cur = [], []
        for i in touching:
            g = work[i]
            single = (g['op'] == 'R' and sum(a != 'I' for a in g['pauli']) == 1 and not g['da']
                      and len(g['th']) == 1 and all(count[k] == 1 for k in g['th']))
            if single:
                cur.append(i)
            else:
                if cur:
                    runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        for run in runs:
            if len(run) < 3:
                continue
            blk = run[-3:]
            paulis = [work[i]['pauli'][q] for i in blk]
            if paulis[0] == paulis[1] or paulis[1] == paulis[2]:
                continue
            for Q in 'XYZ':
                word = 'I' * q + Q + 'I' * (st.n - q - 1)
                tr = transport(word, work[blk[-1] + 1:])
                if tr is None or not joint_invariant(st, Om, tr[1]):
                    continue
                fld = euler_field(paulis, Q)
                if fld is None:
                    continue
                syms = [next(iter(work[i]['th'])) for i in blk]
                coefs = [work[i]['th'][k] for i, k in zip(blk, syms)]
                halfm = [work[i]['m'][k] for i, k in zip(blk, syms)]
                out.append(dict(qubit=q, source_gates=[work[i]['src'] for i in blk], work_positions=blk, block_paulis=paulis,
                                symbols=syms, coefficients=[str(c) for c in coefs], half_angle_m=halfm,
                                inserted_pauli=Q, output_generator=tr[1], output_sign=tr[0],
                                field_alpha_coordinates=fld['w_text'], cramer_determinant=fld['det_text'],
                                euler_identity_verified_by_groebner_reduction=True))
    return out


def field_vectors(st, point, inactive, vfields):
    """Exact family vectors at theta* (Fractions)."""
    vecs, labels = [], []
    for k in inactive:
        v = [F(0)] * st.K; v[k] = F(1)
        vecs.append(v); labels.append('e_%d (inactive symbol)' % k)
    for f in vfields:
        fld = euler_field(f['block_paulis'], f['inserted_pauli'])
        (C, S) = fld['symbols']
        subs = {}
        for j, (k, m) in enumerate(zip(f['symbols'], f['half_angle_m'])):
            c, s = half_angle_cs(point, k, m)
            subs[C[j]] = sp.Rational(c.numerator, c.denominator)
            subs[S[j]] = sp.Rational(s.numerator, s.denominator)
        v = [F(0)] * st.K
        for j, (k, c) in enumerate(zip(f['symbols'], f['coefficients'])):
            val = sp.Rational(fld['w'][j].subs(subs))
            v[k] = F(int(val.p), int(val.q)) / F(c)
        vecs.append(v)
        labels.append('virtual %s after qubit-%d block %s' % (f['inserted_pauli'], f['qubit'], f['symbols']))
    return vecs, labels


# ----------------------------------------------------------------------------
# Symbolic two-copy Gram expansion (exact, budget limited)
# ----------------------------------------------------------------------------
def _padd(a, b, sign):
    d = dict(a)
    for v, c in b:
        d[v] = d.get(v, F(0)) + sign * c
        if not d[v]:
            del d[v]
    return tuple(sorted(d.items()))


def symbolic_two_copy(st, work, time_budget=60.0, term_budget=1500000, project=True):
    """K(x,x';theta) = Tr[(rho(x)(x)rho(x')) Omega] by two-copy Heisenberg propagation.

    Variables: theta_k -> k; copy-1 base b -> K+b; copy-2 base b -> K+B+b (full angles).
    With project=False the propagated two-copy operator U^dag(x)2 Omega U(x)2 of the
    given gate list is returned grouped by Pauli word (no expectation value taken).
    """
    t0 = time.perf_counter()
    n, K, B = st.n, st.K, st.B
    terms = {(o + o, ()): (F(w), F(0)) for o, w in zip(st.obs, st.w)}
    peak = len(terms)
    for g in reversed(work):
        if g['op'] != 'R':
            new = {}
            for (word, ph), c in terms.items():
                s1, w1 = ex.conjugate(word[:n], dict(op=g['op'], wires=list(g['wires'])))
                s2, w2 = ex.conjugate(word[n:], dict(op=g['op'], wires=list(g['wires'])))
                ex.push(new, (w1 + w2, ph), ex.scale(c, F(s1 * s2)))
            terms = new
            continue
        for copy in (0, 1):
            gen = ('I' * n + g['pauli']) if copy else (g['pauli'] + 'I' * n)
            if 'fresh' in g:   # deletion check: the gate's angle replaced by a fresh variable
                angle = tuple(sorted([(g['fresh'][copy], F(1))] + ([(K + copy * B + b, c) for b, c in g['da'].items()]
                                                                if g['fresh_keep_data'] else [])))
            else:
                angle = tuple(sorted([(k, c) for k, c in g['th'].items()] +
                                     [(K + copy * B + b, c) for b, c in g['da'].items()]))
            new = {}
            for (word, ph), c in terms.items():
                if ex.commutes(word, gen):
                    ex.push(new, (word, ph), c)
                    continue
                z, w2 = ex.product(gen, word)
                for direction in (-1, 1):
                    p2 = _padd(ph, angle, direction)
                    ex.push(new, (word, p2), ex.scale(c, F(1, 2)))
                    ex.push(new, (w2, p2), ex.scale(ex.mul(c, z), F(direction, 2)))
            terms = new
            peak = max(peak, len(terms))
            if len(terms) > term_budget:
                return dict(completed=False, reason='term budget %d exceeded' % term_budget,
                            seconds=time.perf_counter() - t0, peak_terms=peak)
            if time.perf_counter() - t0 > time_budget:
                return dict(completed=False, reason='time budget %.0f s exceeded' % time_budget,
                            seconds=time.perf_counter() - t0, peak_terms=peak)
    if not project:
        grouped = {}
        for (word, ph), c in terms.items():
            grouped.setdefault(word, {})[ph] = c
        return dict(completed=True, seconds=time.perf_counter() - t0, peak_terms=peak, operator=grouped)
    result = {}
    for (word, ph), c in terms.items():
        if set(word) <= set('IZ'):
            ex.push(result, ph, c)
    block = {}
    for ph, c in result.items():
        nu = tuple((v, x) for v, x in ph if v < K)
        w1 = tuple((v, x) for v, x in ph if K <= v < K + B)
        w2 = tuple((v, x) for v, x in ph if v >= K + B)
        if w1 and w2:
            block.setdefault((w1, w2), {})
            ex.push(block[(w1, w2)], nu, c)
    block = {k: v for k, v in block.items() if v}
    dependent = sorted({v for entry in block.values() for nu in entry for v, x in nu})
    return dict(completed=True, seconds=time.perf_counter() - t0, peak_terms=peak,
                kernel_terms=len(result), gram_block_entries=len(block),
                inactive=sorted(set(range(K)) - set(dependent)), block=block)


def field_trig_poly(field):
    """Virtual field v_k(theta) as exact exponential sums {freq tuple over symbols: Gaussian coeff}."""
    fld = euler_field(field['block_paulis'], field['inserted_pauli'])
    C, S = fld['symbols']
    Z = sp.symbols('Z0:3')
    out = {}
    for j, (k, c) in enumerate(zip(field['symbols'], field['coefficients'])):
        e = fld['w'][j]
        for jj in range(3):
            e = e.subs({C[jj]: (Z[jj] + 1 / Z[jj]) / 2, S[jj]: (Z[jj] - 1 / Z[jj]) / (2 * sp.I)})
        e = sp.expand(e / sp.Rational(str(F(c))))
        terms = {}
        for term in sp.Add.make_args(e):
            coeff, mono = term.as_coeff_Mul()
            powers = mono.as_powers_dict()
            ic = sp.Integer(1)
            freq = {}
            for base, ex_ in powers.items():
                if base == sp.I:
                    ic *= sp.I ** ex_
                elif base in Z:
                    jj = Z.index(base)
                    # Z_jj = exp(i c_jj theta_k / 2)
                    kk = field['symbols'][jj]
                    freq[kk] = freq.get(kk, F(0)) + F(int(ex_)) * F(field['coefficients'][jj]) / 2
                elif base != 1:
                    ic *= base ** ex_
            val = sp.nsimplify(coeff * ic)
            re_, im_ = sp.re(val), sp.im(val)
            key = tuple(sorted((a, b) for a, b in freq.items() if b))
            ex.push(terms, key, (F(str(re_)), F(str(im_))))
        out[k] = terms
    return out


def verify_field_on_symbolic_gram(block, vfield_terms):
    """Exact check that sum_k v_k(theta) d_k Ghat(theta) == 0 for every Gram entry."""
    for entry in block.values():
        acc = {}
        for k, vt in vfield_terms.items():
            for nu, c in entry.items():
                nk = dict(nu).get(k, F(0))
                if not nk:
                    continue
                dc = ex.mul(c, (F(0), nk))   # d/d theta_k e^{i nu theta} = i nu_k e^{...}
                for mu, a in vt.items():
                    ex.push(acc, _padd(nu, mu, 1), ex.mul(dc, a))
        if acc:
            return False
    return True


def _support(g):
    if g['op'] == 'R':
        return {q for q, a in enumerate(g['pauli']) if a != 'I'}
    return set(g['wires'])


def _gates_commute(a, b):
    if a['op'] == 'R' and b['op'] == 'R':
        return ex.commutes(a['pauli'], b['pauli'])   # Pauli rotations, any angles
    return not (_support(a) & _support(b))


def light_cone(suffix):
    """Drop later gates that commute with the first gate and with every kept gate.

    Such a gate can be moved in front of the first gate (exact operator identity),
    i.e. into the arbitrary prefix, so the two-copy suffix operator identity is unchanged."""
    kept = [suffix[0]]
    for h in suffix[1:]:
        if not all(_gates_commute(h, k) for k in kept):
            kept.append(h)
    return kept


def deletion_operator_check(st, entry, time_budget=60.0, term_budget=600000):
    """Independent exact check of one deletion step.

    The deleted gate's angle (theta part for the joint rule, whole angle with one fresh
    variable per copy for the commutant rule) is replaced by fresh variables and the
    two-copy operator of the gate suffix is expanded symbolically; the step is confirmed
    iff no term depends on the fresh variables."""
    suffix = light_cone(entry['_suffix'])
    res = symbolic_two_copy(st, suffix, time_budget, term_budget, project=False)
    if not res['completed']:
        return dict(completed=False, reason=res['reason'], seconds=res['seconds'], light_cone_gates=len(suffix))
    tau = {st.K + 2 * st.B, st.K + 2 * st.B + 1}
    dep = any(v in tau for terms in res['operator'].values() for ph in terms for v, _ in ph)
    return dict(completed=True, fresh_angle_independent=not dep, suffix_gates=len(entry['_suffix']),
                light_cone_gates=len(suffix),
                operator_words=len(res['operator']), seconds=res['seconds'])


def suffix_operator_check(st, work, field, time_budget=60.0, term_budget=600000):
    """Independent exact check of a virtual field: sum_k v_k d_k Omega_suffix(theta) == 0.

    Omega_suffix = (U_s (x) U_s)^dag Omega (U_s (x) U_s) for the gate suffix starting at
    the Euler block; K = Tr[(rho_prefix (x) rho_prefix') Omega_suffix] is linear in it,
    so the operator identity implies J(theta) v(theta) = 0 for all theta and inputs.
    A perturbed field is checked as a negative control (must fail)."""
    start = field['work_positions'][0]
    suffix = light_cone(work[start:])
    res = symbolic_two_copy(st, suffix, time_budget, term_budget, project=False)
    if not res['completed']:
        return dict(completed=False, reason=res['reason'], seconds=res['seconds'])
    vt = field_trig_poly(field)
    ok = verify_field_on_symbolic_gram(res['operator'], vt)
    k0 = field['symbols'][1]
    bad = dict(vt)
    bad[k0] = {key: ex.scale(c, F(2)) for key, c in vt[k0].items()}
    control = verify_field_on_symbolic_gram(res['operator'], bad)
    return dict(completed=True, identity_holds=ok, perturbed_field_rejected=not control,
                suffix_gates=len(work) - start, light_cone_gates=len(suffix), operator_words=len(res['operator']),
                seconds=res['seconds'])


def span_rank(block, max_cells=4000000):
    """dim_C span of non-constant parts of the Gram entries (exact, over Q(i))."""
    cols = sorted({nu for entry in block.values() for nu in entry if nu})
    rows = [entry for entry in block.values()]
    if len(rows) * len(cols) * 4 > max_cells:
        return None, len(rows), len(cols)
    ci = {nu: j for j, nu in enumerate(cols)}
    m = len(cols)
    real_rows = []
    for entry in rows:
        re_row, im_row = [F(0)] * (2 * m), [F(0)] * (2 * m)
        for nu, (a, b) in entry.items():
            if not nu:
                continue
            j = ci[nu]
            re_row[j], re_row[m + j] = F(a), -F(b)
            im_row[j], im_row[m + j] = F(b), F(a)
        real_rows += [re_row, im_row]
    r, _, _ = rank_exact(real_rows, 2 * m)
    assert r % 2 == 0
    return r // 2, len(rows), len(cols)
