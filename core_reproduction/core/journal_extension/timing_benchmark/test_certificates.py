import unittest
import numpy as np
from structural_core import Gate, rotation, pauli, h, cx, states_and_tangent
from certificates import reduce_circuit, projected_kernel, invariant_generator, rbf_output_derivative_bound


def xyz(n):
    return [pauli(n, a, q) for q in range(n) for a in 'XYZ']


def dense_states(circuit, X, theta, n):
    """Independent small dense implementation; no structural_core apply calls."""
    matrices = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]),
                'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.diag([1, -1])}
    def tensor(word):
        out = np.ones((1, 1))
        for a in word:
            out = np.kron(out, matrices[a])
        return out
    result = []
    for x in X:
        v = np.zeros(2 ** n, complex)
        v[0] = 1
        for g in circuit:
            if g.op == 'R':
                angle = g.value if g.kind == 'fixed' else x[g.feature] if g.kind == 'data' else theta[g.index] * (x[g.feature] if g.kind == 'product' else 1)
                U = np.cos(angle/2) * np.eye(2**n) - 1j * np.sin(angle/2) * tensor(g.pauli)
            elif g.op == 'H':
                q = g.wires[0]
                U = (tensor(pauli(n, 'X', q)) + tensor(pauli(n, 'Z', q))) / np.sqrt(2)
            else:
                c, t = g.wires
                U = np.zeros((2**n, 2**n))
                for col in range(2**n):
                    bits = list(format(col, '0%db' % n))
                    if bits[c] == '1':
                        bits[t] = '0' if bits[t] == '1' else '1'
                    U[int(''.join(bits), 2), col] = 1
            v = U @ v
        result.append(v)
    return np.asarray(result)


class CertificateTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(4).uniform(-2, 2, (11, 2))
        self.prefix = [rotation(2, 'Y', 0, 'data', feature=0), rotation(2, 'X', 1, 'data', feature=1)]
        self.obs = xyz(2)

    def assert_kernel_equal(self, circuit, obs, w, method='transport_metric'):
        reduced, report = reduce_circuit(circuit, obs, w, method)
        theta = np.array([.71, -.31, .44, .23])
        for kind in ('dot', 'rbf'):
            a = projected_kernel(circuit, self.X, theta, 2, obs, w, kind)[0]
            b = projected_kernel(reduced, self.X, theta, 2, obs, w, kind)[0]
            np.testing.assert_allclose(a, b, atol=1e-12)
        return report

    def test_xyz_changes_features_not_kernel(self):
        circuit = self.prefix + [rotation(2, 'Z', 0, 'param')]
        report = self.assert_kernel_equal(circuit, self.obs, np.ones(6))
        self.assertEqual(report['inactive_parameters'], [0])
        a = projected_kernel(circuit, self.X, [.7], 2, self.obs, np.ones(6))[2]
        b = projected_kernel(circuit, self.X, [0], 2, self.obs, np.ones(6))[2]
        self.assertGreater(np.max(np.abs(a-b)), .1)

    def test_z_only_not_fidelity_suffix(self):
        c = self.prefix + [rotation(2, 'Y', 0, 'param')]
        _, r = reduce_circuit(c, ['ZI'], [1])
        self.assertEqual(r['inactive_parameters'], [])
        d = projected_kernel(c, self.X, [.7], 2, ['ZI'], [1], parameter=0)[1]
        self.assertGreater(np.max(np.abs(d)), .01)

    def test_anisotropy_unknown(self):
        ok, _ = invariant_generator('ZI', self.obs, [1, 1.01, 1, 1, 1, 1])
        self.assertFalse(ok)

    def test_no_tolerance_certificate(self):
        ok, _ = invariant_generator('ZI', self.obs, [1, 1+1e-12, 1, 1, 1, 1])
        self.assertFalse(ok)

    def test_entangling_missing_partner(self):
        self.assertFalse(invariant_generator('ZZ', self.obs, np.ones(6))[0])

    def test_entangling_closed_readout(self):
        c = self.prefix + [Gate('R', 'ZZ', 'param')]
        r = self.assert_kernel_equal(c, ['XI', 'YZ', 'ZI'], [1, 1, 2])
        self.assertEqual(r['inactive_parameters'], [0])

    def test_clifford_transport(self):
        c = self.prefix + [rotation(2, 'Z', 0, 'param'), h(0), cx(0, 1)]
        # ZI -> XI -> XX; ZI pairs with YX under XX.
        r = self.assert_kernel_equal(c, ['ZI', 'YX'], [1, 1])
        self.assertEqual(r['inactive_parameters'], [0])

    def test_commuting_data_transport(self):
        c = self.prefix + [rotation(2, 'Z', 0, 'param'), rotation(2, 'Z', 0, 'data')]
        self.assertEqual(self.assert_kernel_equal(c, self.obs, np.ones(6))['inactive_parameters'], [0])

    def test_shared_parameter_not_globally_dead(self):
        c = [rotation(2, 'Y', 0, 'param')] + self.prefix + [rotation(2, 'X', 0, 'data'), rotation(2, 'Z', 0, 'param')]
        r = self.assert_kernel_equal(c, self.obs, np.ones(6))
        self.assertEqual(len(r['certificates']), 1)
        self.assertEqual(r['inactive_parameters'], [])
        self.assertEqual(r['unknown_parameters'], [0])

    def test_shared_all_removed(self):
        c = self.prefix + [rotation(2, 'X', 0, 'param'), rotation(2, 'Y', 1, 'param')]
        r = self.assert_kernel_equal(c, self.obs, np.ones(6))
        self.assertEqual(r['inactive_parameters'], [0])

    def test_product_never_common(self):
        c = self.prefix + [rotation(2, 'Z', 0, 'product')]
        self.assertEqual(reduce_circuit(c, self.obs, np.ones(6))[1]['inactive_parameters'], [])

    def test_continuation_not_prunable(self):
        c = self.prefix + [rotation(2, 'Z', 0, 'param')]
        self.assertEqual(reduce_circuit(c, ['ZI'], [1])[1]['inactive_parameters'], [0])
        c += [rotation(2, 'Y', 0, 'data')]
        self.assertEqual(reduce_circuit(c, ['ZI'], [1])[1]['inactive_parameters'], [])
        self.assertGreater(np.max(np.abs(projected_kernel(c, self.X, [.7], 2, ['ZI'], [1], parameter=0)[1])), .01)

    def test_dense_independent_simulator(self):
        rng = np.random.default_rng(92)
        for _ in range(10):
            c = self.prefix + [h(0), cx(0, 1)]
            for j in range(4):
                c.append(Gate('R', ''.join(rng.choice(list('IXYZ'), 2)), ['param', 'product', 'fixed', 'data'][j], j, j % 2, .37))
            theta = rng.normal(size=4)
            np.testing.assert_allclose(states_and_tangent(c, self.X, theta, 2)[0], dense_states(c, self.X, theta, 2), atol=1e-12)

    def test_analytic_gradient_shared_vs_finite_difference(self):
        c = self.prefix + [rotation(2, 'Y', 0, 'product'), cx(0,1), rotation(2, 'X', 1, 'param')]
        for kind in ('dot', 'rbf'):
            d = projected_kernel(c, self.X, [.7], 2, self.obs, np.arange(1,7), kind, parameter=0)[1]
            a = projected_kernel(c, self.X, [.7+1e-6], 2, self.obs, np.arange(1,7), kind)[0]
            b = projected_kernel(c, self.X, [.7-1e-6], 2, self.obs, np.arange(1,7), kind)[0]
            np.testing.assert_allclose(d, (a-b)/2e-6, atol=2e-8)

    def test_bound_closed_and_symmetric(self):
        self.assertEqual(rbf_output_derivative_bound('ZI', self.obs, np.ones(6)), 0)
        with self.assertRaises(ValueError):
            rbf_output_derivative_bound('ZZ', self.obs, np.ones(6))

    def test_reject_invalid_metric(self):
        for weights in ([1,0], [1,-1], [1,np.nan], [1,np.inf]):
            with self.assertRaises(ValueError):
                invariant_generator('Z', ['X','Y'], weights)


if __name__ == '__main__':
    unittest.main()
