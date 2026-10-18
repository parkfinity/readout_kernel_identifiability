"""Unit tests: n = 2, 4 reproduction, engine agreement (tol 1e-10), exact algebra, negative controls."""
import copy
import itertools
import json
import math
import random
import unittest
from fractions import Fraction as F

import numpy as np

import kyriienko_scaling as ks

try:
    import qiskit
    QISKIT_124 = qiskit.__version__ == '1.2.4'
except Exception:  # pragma: no cover
    QISKIT_124 = False

TOL = 1e-10


def frozen(n, kind):
    p = ks.FROZEN / ('%s_n%d_l1_%s.json' % (ks.MODULE, n, kind))
    return json.loads(p.read_text())


def info_for(n, kind):
    """Frozen IR for n = 2, 4 (no Qiskit needed); source route for larger n."""
    return frozen(n, kind) if n in (2, 4) else ks.make_info(n, kind)


def exact_d2(info, theta_u, s, sp):
    eng = ks.ExactEngine(info)
    a = eng.point(theta_u, s, info['observables'])
    b = eng.point(theta_u, sp, info['observables'])
    idx = np.arange(len(info['observables']))
    rows, scales, d2 = ks.pair_rows([a, b], idx, info['weights'], [(0, 1)])
    grad = [scales[0] * v for v in rows[0]]
    return d2[0], grad


class TestProvenance(unittest.TestCase):
    def test_core_copy_identical(self):
        p = ks.provenance()
        self.assertTrue(p['copied_core_identical'])
        self.assertGreaterEqual(len(p['copied_core']), 30)

    @unittest.skipUnless(QISKIT_124, 'needs Qiskit 1.2.4 (frozen census compiler)')
    def test_frozen_ir_reproduced_n2_n4(self):
        for n in (2, 4):
            for kind in ks.READOUTS:
                info, chk = ks.gate_ir_check(n, kind)
                self.assertTrue(chk['dict_equal'], chk)
                self.assertTrue(chk['sha_equal'], chk)

    @unittest.skipUnless(QISKIT_124, 'needs Qiskit 1.2.4')
    def test_source_formula_and_kyriform_all_sizes(self):
        for n in (2, 4, 6, 8):
            for kind in ks.READOUTS:
                s = ks.structure_checks(ks.make_info(n, kind))
                for key in ('formula_equivalent', 'fixed_frequency', 'kyriform_euler_blocks',
                            'each_parameter_once', 'entangler_commutes_Z0', 'entangler_commutes_X_last',
                            'unit_weights'):
                    self.assertTrue(s[key], (n, kind, key))
                kap = {k['qubit']: (k['feature'], k['kappa']) for k in s['kappa']}
                for q in range(n):  # kappa_q = 4 floor(q/2) + n + 4, feature q mod 2 (n even)
                    self.assertEqual(kap[q], ([q % 2], 4 * (q // 2) + n + 4))

    def test_formula_check_rejects_modified_ir(self):
        info = copy.deepcopy(frozen(4, 'local_xyz'))
        info['data_expressions'][1] = '7.0*acos(a0)'
        self.assertFalse(ks.structure_checks(info)['formula_equivalent'])
        info = copy.deepcopy(frozen(4, 'local_xyz'))
        cx = [g for g in info['program'] if g['op'] == 'CX']
        cx[-1]['wires'] = [2, 1]
        s = ks.structure_checks(info)
        self.assertFalse(s['formula_equivalent'])


class TestReproduceKnownRanks(unittest.TestCase):
    """Exact certification on the frozen n = 2, 4 IR reproduces the closed values."""
    EXPECTED = {(2, 'local_xyz'): 4, (4, 'local_xyz'): 10, (4, 'pauli_weight_le_2'): 10,
                (2, 'local_z'): 4, (4, 'local_z'): 8, (2, 'pauli_weight_le_2'): 0}

    def test_ranks(self):
        for n in (2, 4):
            cache = ks.PointCache(frozen(n, 'pauli_weight_le_2'))
            for kind in ks.READOUTS:
                info = dict(frozen(n, kind), _kind=kind)
                r = ks.certify_setting(info, cache, numeric=True, name='test', log=lambda *a: None)
                exp = self.EXPECTED[n, kind]
                self.assertEqual(r['certified_lower_bound']['value'], exp, (n, kind))
                self.assertEqual(r['upper_bound']['value'], exp, (n, kind))
                self.assertEqual(r['certified_r'], exp)
                self.assertEqual(r['numerical_not_certified']['rank'], exp)
                self.assertTrue(r['certified_lower_bound']['sympy_domainmatrix_det_equal'])
                for p, v in r['certified_lower_bound']['residues'].items():
                    self.assertEqual(v['det_mod_p'], v['bareiss_mod_p'])
                    if exp:
                        self.assertNotEqual(v['det_mod_p'], 0)
                self.assertTrue(all(r['exact_consistency_at_theta_star']['null_vectors_annihilate_all_rows']))


class TestEngines(unittest.TestCase):
    def test_exact_d2_vs_independent_dense_random(self):
        rng = random.Random(7)
        for n in (2, 4):
            for kind in ks.READOUTS:
                info = info_for(n, kind)
                for _ in range(3):
                    u = [F(rng.randint(-40, 40), rng.randint(1, 41)) for _ in range(info['count'])]
                    s = [F(rng.randint(1, 30), rng.randint(1, 30)) for _ in range(2)]
                    sp = [F(rng.randint(1, 30), rng.randint(1, 30)) for _ in range(2)]
                    d2, _ = exact_d2(info, u, s, sp)
                    th = [ks.theta_from_u(v) for v in u]
                    x = [float(ks.x_from_s(v)) for v in s]
                    xp = [float(ks.x_from_s(v)) for v in sp]
                    dd, _, _ = ks.dense_d2(n, th, x, xp, info['observables'], info['weights'])
                    self.assertLess(abs(float(d2) - dd), TOL, (n, kind))

    @unittest.skipUnless(QISKIT_124, 'needs Qiskit 1.2.4')
    def test_exact_d2_vs_dense_n6(self):
        info = ks.make_info(6, 'local_xyz')
        u = [F(k + 3, 2 * k + 7) for k in range(18)]
        d2, _ = exact_d2(info, u, [F(2, 7), F(5, 3)], [F(9, 4), F(1, 8)])
        dd, _, _ = ks.dense_d2(6, [ks.theta_from_u(v) for v in u], [float(ks.x_from_s(F(2, 7))), float(ks.x_from_s(F(5, 3)))],
                               [float(ks.x_from_s(F(9, 4))), float(ks.x_from_s(F(1, 8)))], info['observables'], info['weights'])
        self.assertLess(abs(float(d2) - dd), TOL)

    @unittest.skipUnless(QISKIT_124, 'needs Qiskit 1.2.4')
    def test_float_engine_vs_dense_vs_qiskit_random(self):
        rng = np.random.default_rng(3)
        for n in (2, 4, 6, 8):
            for kind in ks.READOUTS:
                info = ks.make_info(n, kind)
                eng = ks.FloatEngine(info)
                for _ in range(2):
                    th = rng.uniform(-np.pi, np.pi, 3 * n)
                    X = rng.uniform(-1, 1, (2, 2))
                    d2, _ = eng.jacobian(th, X[:1], X[1:], info['observables'], info['weights'])
                    dd, fa, fb = ks.dense_d2(n, th, X[0], X[1], info['observables'], info['weights'])
                    fq = ks.qiskit_source_features(n, th, X, info['observables'])
                    dq = float(np.sum((fq[0] - fq[1]) ** 2))
                    self.assertLess(abs(d2[0] - dd), TOL, (n, kind))
                    self.assertLess(abs(dq - dd), TOL, (n, kind))
                    self.assertLess(float(np.max(np.abs(fq[0] - fa))), TOL)

    def test_exact_gradient_vs_finite_difference(self):
        info = frozen(4, 'pauli_weight_le_2')
        u = [F(k + 2, k + 9) for k in range(12)]
        s, sp = [F(1, 3), F(5, 2)], [F(4, 7), F(3, 2)]
        d2, grad = exact_d2(info, u, s, sp)
        th = np.array([ks.theta_from_u(v) for v in u])
        x = [float(ks.x_from_s(v)) for v in s]
        xp = [float(ks.x_from_s(v)) for v in sp]
        h = 1e-5
        for k in range(12):
            e = np.zeros(12)
            e[k] = h
            fd = (ks.dense_d2(4, th + e, x, xp, info['observables'], info['weights'])[0]
                  - ks.dense_d2(4, th - e, x, xp, info['observables'], info['weights'])[0]) / (2 * h)
            self.assertLess(abs(fd - float(grad[k])), 1e-7, k)
        self.assertEqual(grad[2], 0)  # theta_2: exact zero column (Z_0 output gauge)


class TestExactAlgebra(unittest.TestCase):
    def test_euler_identity(self):
        self.assertTrue(ks.euler_identity_symbolic()['passed'])

    def test_two_copy_invariance_and_controls(self):
        obs, w = ks.readout(4, 'local_xyz')
        self.assertTrue(ks.two_copy_invariant(obs, w, 'ZIII'))
        self.assertTrue(ks.two_copy_invariant(obs, w, 'IIIX'))
        self.assertFalse(ks.two_copy_invariant(obs, w, 'ZZII'))
        aniso = [1.0] * 4 + [2.0] * 4 + [3.0] * 4
        self.assertFalse(ks.two_copy_invariant(obs, aniso, 'ZIII'))
        obs2, w2 = ks.readout(4, 'pauli_weight_le_2')
        self.assertTrue(ks.two_copy_invariant(obs2, w2, 'IIIX'))
        self.assertFalse(ks.two_copy_invariant(obs2, w2, 'XXXI'))
        obsz, wz = ks.readout(4, 'local_z')
        self.assertTrue(ks.two_copy_invariant(obsz, wz, 'ZZZI'))
        self.assertFalse(ks.two_copy_invariant(obsz, wz, 'IIIX'))

    def test_anisotropic_readout_loses_gauges(self):
        info = copy.deepcopy(frozen(4, 'local_xyz'))
        info['weights'] = [1.0] * 4 + [2.0] * 4 + [3.0] * 4
        ub = ks.upper_bound_certificate(info, ks.theta_star(12))
        self.assertEqual(ub['upper'], 12)

    def test_linear_algebra(self):
        rng = random.Random(5)
        for m in (1, 3, 6, 9):
            M = [[rng.randint(-10 ** 30, 10 ** 30) for _ in range(m)] for _ in range(m)]
            d = ks.bareiss_det(M)
            self.assertEqual(d, ks.sympy_det(M))
            for p in ks.PRIMES:
                self.assertEqual(ks.det_mod(M, p), d % p)
        M = [[1, 2, 3], [2, 4, 6], [1, 0, 1]]
        self.assertEqual(ks.bareiss_det(M), 0)
        rows = [[0, 0, 0], [1, 2, 3], [2, 4, 6], [0, 1, 1], [5, 5, 5]]
        r, rs, cs = ks.select_minor_mod(rows, ks.PRIMES[0])
        self.assertEqual(r, 3)
        sub = [[rows[i][k] for k in sorted(cs)] for i in sorted(rs)]
        self.assertNotEqual(ks.bareiss_det(sub), 0)

    def test_gunit(self):
        for r in (F(2, 5), F(-3, 7), F(9, 4)):
            A, B, C = ks.gunit(r)
            self.assertEqual(A * A + B * B, C * C)
            ang = 2 * math.atan(float(r))
            self.assertAlmostEqual(A / C, math.cos(ang), 14)
            self.assertAlmostEqual(B / C, math.sin(ang), 14)


if __name__ == '__main__':
    unittest.main()
