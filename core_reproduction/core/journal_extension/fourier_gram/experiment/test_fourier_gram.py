"""Unit tests for the Fourier-Gram certification (python3 -m unittest -v)."""
from fractions import Fraction as F
import json
import math
import sys
import unittest

import numpy as np

import fourier_gram as fg
sys.path.insert(0, str(fg.HERE / 'copied_from_extension_v1'))
import interval_engine as ie  # noqa: E402
from mpmath import iv  # noqa: E402

SRC = fg.HERE.parents[1] / 'certified_dimension_extension_v1' / 'frozen_inputs'


def info(name):
    return json.loads((SRC / (name + '.json')).read_text())


def load(name):
    return fg.Setting(info(name), name)


HUB_Z = 'hubregtsen_encoding_circuit_n4_l1_local_z'
KYR2_XYZ = 'kyriienko_nonlinear_encoding_circuit_n2_l1_local_xyz'
KYR2_W2 = 'kyriienko_nonlinear_encoding_circuit_n2_l1_pauli_weight_le_2'
KYR4_XYZ = 'kyriienko_nonlinear_encoding_circuit_n4_l1_local_xyz'


class FourierGramTests(unittest.TestCase):
    def test_hubregtsen_gram_is_beta_diagonal_and_rank_two(self):
        st = load(HUB_Z)
        pt = fg.pythagorean_point(st, 3)
        C, dC, labels = fg.LaurentEngine(st, pt, 'exact').run().coefficients_exact()
        A = len(labels)
        G = [[sum(C[o][a] * C[o][b] for o in range(len(st.obs))) for b in range(A)] for a in range(A)]
        sin = [fg.half_angle_cs(pt, q, st.L[q])[1] for q in range(4)]   # exact sin(theta_q)
        beta = {('cos', (2, 0)): sin[0] ** 2 + sin[2] ** 2, ('cos', (0, 2)): sin[1] ** 2 + sin[3] ** 2}
        for a in range(A):
            for b in range(A):
                want = beta.get(labels[a], F(0)) if a == b else F(0)
                self.assertEqual(G[a][b], want, (labels[a], labels[b]))
        # cos(x_j) = cos(2 psi_j) because psi = x/2 for raw data bases
        self.assertEqual(st.M, [2, 2])
        J, _, _ = fg.gram_jacobian_mod(*fg.LaurentEngine(st, pt, 'mod').run().coefficients_mod()[:2], st.w, fg.PRIMES[0])
        self.assertEqual(fg.rank_mod(J, fg.PRIMES[0])[0], 2)
        work, inactive, _ = fg.certified_reduction(st)
        self.assertEqual(inactive, [4, 5, 6, 7])
        res = fg.symbolic_two_copy(st, st.gates)
        self.assertTrue(res['completed'])
        self.assertEqual(fg.span_rank(res['block'])[0], 2)

    def test_distance_gram_matches_statevector(self):
        for name in (KYR4_XYZ, 'yz_cx_encoding_circuit_n2_l2_local_xyz', 'hubregtsen_encoding_circuit_n2_l2_pauli_weight_le_2'):
            st = load(name)
            eng = fg.NumericEngine(st)
            rng = np.random.default_rng(5)
            for _ in range(4):
                th = rng.uniform(-3, 3, st.K)
                C, _ = eng.coefficients(th)
                x, xp = rng.uniform(-0.99, 0.99, 2), rng.uniform(-0.99, 0.99, 2)
                self.assertLess(abs(eng.distance_gram(C, x, xp) - eng.distance_direct(th, x, xp)), 1e-10)

    def test_known_inactive_symbol_has_vanishing_gram_derivative(self):
        st = load(HUB_Z)   # controlled-Z rotations 4..7 commute with every measured Z
        res = fg.symbolic_two_copy(st, st.gates)
        self.assertTrue(set([4, 5, 6, 7]) <= set(res['inactive']))
        self.assertTrue(set(res['inactive']).isdisjoint({0, 1, 2, 3}))
        pt = fg.pythagorean_point(st, 4)
        J, _, _ = fg.gram_jacobian_mod(*fg.LaurentEngine(st, pt, 'mod').run().coefficients_mod()[:2], st.w, fg.PRIMES[0])
        self.assertTrue(all((J[:, k] == 0).all() for k in (4, 5, 6, 7)))
        self.assertTrue(all((J[:, k] != 0).any() for k in (0, 1, 2, 3)))
        st2 = load(KYR2_W2)  # complete 2-qubit Pauli readout: Gram independent of theta
        res2 = fg.symbolic_two_copy(st2, st2.gates)
        self.assertEqual(res2['inactive'], list(range(6)))

    def test_exact_modular_numeric_coefficients_agree(self):
        st = load(KYR2_XYZ)
        pt = fg.pythagorean_point(st, 9)
        Ce, dCe, labels = fg.LaurentEngine(st, pt, 'exact').run().coefficients_exact()
        p = fg.PRIMES[1]
        Cm, dCm, ok = fg.LaurentEngine(st, pt, 'mod', p).run().coefficients_mod()
        self.assertTrue(ok)
        Cn, dCn = fg.NumericEngine(st).coefficients(np.array(pt['theta_float']))
        col = {lab: j for j, lab in enumerate(st.basis)}
        for o in range(len(st.obs)):
            for j, lab in enumerate(labels):
                self.assertEqual(fg.frac_mod(Ce[o][j], p), Cm[o, col[lab]])
                self.assertLess(abs(float(Ce[o][j]) - Cn[o, col[lab]]), 1e-12)
                for k in range(st.K):
                    self.assertEqual(fg.frac_mod(dCe[k][o][j], p), dCm[k, o, col[lab]])
                    self.assertLess(abs(float(dCe[k][o][j]) - dCn[k, o, col[lab]]), 1e-12)

    def test_euler_field_closed_form(self):
        fld = fg.euler_field(['Z', 'X', 'Z'], 'X')
        C, S = fld['symbols']
        for (a, b, c) in [(3, 4, 5), (5, 12, 13), (8, 15, 17)]:
            vals = {C[0]: F(a, c), S[0]: F(b, c), C[1]: F(b, c), S[1]: F(-a, c), C[2]: F(-a, c), S[2]: F(b, c)}
            full = [(vals[C[j]] ** 2 - vals[S[j]] ** 2, 2 * vals[S[j]] * vals[C[j]]) for j in range(3)]
            w = [F(str(e.subs(vals))) for e in fld['w']]
            self.assertEqual(w, [full[2][1], full[2][0] * full[1][1], -full[2][1] * full[1][0]])
            self.assertEqual(F(str(fld['det'].subs(vals))), full[1][1])

    def test_invariance_checks(self):
        st = load(KYR2_XYZ)
        Om = fg.omega_terms(st)
        self.assertTrue(fg.joint_invariant(st, Om, 'IX'))
        self.assertTrue(fg.joint_invariant(st, Om, 'ZI'))
        self.assertFalse(fg.joint_invariant(st, Om, 'XX'))
        self.assertFalse(fg.commutant_invariant(st, Om, 'ZI'))
        self.assertEqual(fg.transport('IX', [dict(op='CX', wires=(0, 1))]), (1, 'IX'))
        self.assertEqual(fg.transport('XI', [dict(op='CX', wires=(0, 1))])[1], 'XX')

    def test_fixed_frequency_detection(self):
        a = info(KYR2_XYZ)
        a = dict(a, program=[dict(a['program'][0], terms=[[1.0, 0, 0]])] + a['program'][1:])
        st = fg.Setting(a)
        self.assertFalse(st.fixed_frequency)
        with self.assertRaises(ValueError):
            fg.LaurentEngine(st, fg.pythagorean_point(st, 1))
        self.assertTrue(load(KYR2_XYZ).fixed_frequency)

    def test_modular_minor_certificate(self):
        p = fg.PRIMES[0]
        J = np.array([[1, 2, 3], [2, 4, 6], [0, 1, 1], [1, 3, 4]], dtype=np.int64)
        r, rows, cols = fg.rank_mod(J, p)
        self.assertEqual(r, 2)
        self.assertNotEqual(fg.det_mod([[J[i, k] for k in cols] for i in rows], p), 0)
        self.assertEqual(fg.rank_exact([[F(int(v)) for v in row] for row in J], 3)[0], 2)

    def test_semantics_match_copied_interval_engine(self):
        iv.dps = 25
        for name in ('hubregtsen_encoding_circuit_n4_l1_local_xyz', 'yz_cx_encoding_circuit_n2_l2_pauli_weight_le_2', KYR4_XYZ):
            inf = info(name)
            st = fg.Setting(inf, name)
            theta = ['%d/17' % (2 + k % 9) for k in range(st.K)]
            x = ['3/19', '-7/19']
            f_iv, _ = ie.features(inf, theta, x)
            phi = np.array([st.base_angles([3 / 19, -7 / 19])])
            f, _ = fg.NumericEngine(st).simulate(np.array([float(F(t)) for t in theta]), phi, tangents=False)
            self.assertLess(max(abs(a - ie.mid(b)) for a, b in zip(f[:, 0], f_iv)), 1e-12)

    def test_symbolic_checks_of_null_fields_and_negative_controls(self):
        st = load(KYR2_XYZ)
        tau = (st.K + 2 * st.B, st.K + 2 * st.B + 1)
        for src, expected in ((3, True), (2, False), (1, False)):   # theta2 inactive; theta1, theta0 active
            i = next(j for j, g in enumerate(st.gates) if g['src'] == src)
            snap = [dict(h) for h in st.gates[i:]]
            snap[0] = dict(st.gates[i], fresh=(tau[0], tau[0]), fresh_keep_data=True)
            chk = fg.deletion_operator_check(st, dict(_suffix=snap))
            self.assertTrue(chk['completed'])
            self.assertEqual(chk['fresh_angle_independent'], expected, src)
        work, inactive, _ = fg.certified_reduction(st)
        (f,) = fg.virtual_fields(st, work)
        chk = fg.suffix_operator_check(st, work, f)
        self.assertTrue(chk['identity_holds'] and chk['perturbed_field_rejected'])

    def test_kyriienko_brackets_close(self):
        for name, want in ((KYR2_XYZ, 4), (KYR4_XYZ, 10)):
            st = load(name)
            pt = fg.pythagorean_point(st, 21)
            C, dC, _ = fg.LaurentEngine(st, pt, 'mod').run().coefficients_mod()
            J, _, _ = fg.gram_jacobian_mod(C, dC, st.w, fg.PRIMES[0])
            self.assertEqual(fg.rank_mod(J, fg.PRIMES[0])[0], want)
            work, inactive, _ = fg.certified_reduction(st)
            vf = fg.virtual_fields(st, work)
            vecs, _ = fg.field_vectors(st, pt, inactive, vf)
            self.assertEqual(st.K - fg.rank_exact(vecs, st.K)[0], want)
            for v in vecs:   # consistency only; the identity itself is structural
                vm = np.array([fg.frac_mod(x, fg.PRIMES[0]) for x in v], dtype=np.int64)
                self.assertTrue((J @ vm % fg.PRIMES[0] == 0).all())


if __name__ == '__main__':
    unittest.main()
