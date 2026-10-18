import unittest
import numpy as np
import run


class WeightThreeTests(unittest.TestCase):
    def test_readout(self):
        words = run.legacy.readout(5, 'weight_le_3')
        self.assertEqual(len(words), 375)
        self.assertEqual(len(set(words)), 375)
        self.assertTrue(all(1 <= sum(c != 'I' for c in w) <= 3 for w in words))

    def test_y_sign(self):
        state = np.array([[1, 1j]]) / np.sqrt(2)
        np.testing.assert_allclose(run.masked_features(state, run.masks(['Y'])), [[1]])

    def test_dense_paulis(self):
        rng = np.random.default_rng(28)
        states = rng.normal(size=(3, 32)) + 1j*rng.normal(size=(3, 32))
        states /= np.linalg.norm(states, axis=1)[:, None]
        words = run.legacy.readout(5, 'weight_le_3')
        reference = np.array([[np.vdot(s, run.dense.pauli_string_op(w) @ s).real
                               for w in words] for s in states])
        np.testing.assert_allclose(run.masked_features(states, run.masks(words)),
                                   reference, atol=1e-14)

    def test_state_normalization(self):
        psi = run.legacy.build_state(5, [.2, -.3], np.arange(15)/7.)
        self.assertAlmostEqual(float(np.linalg.norm(psi)), 1.)

    def test_chain_rule(self):
        lookup = run.masks(run.legacy.readout(2, 'local_xyz'))
        fun = lambda t, x: run.features(t, x, 2, lookup)
        theta = np.arange(6)/3.
        inputs = np.array([[.1, .3], [-.4, .2]])
        _, _, jac = run.distance_and_jacobian(theta, inputs, fun)
        h = 1e-5
        for j in range(6):
            p, m = theta.copy(), theta.copy()
            p[j] += h
            m[j] -= h
            fp, fm = fun(p, inputs), fun(m, inputs)
            value = (np.sum((fp[0]-fp[1])**2)-np.sum((fm[0]-fm[1])**2))/(2*h)
            self.assertAlmostEqual(value, jac[0, j], places=7)

    def test_full_readout_invariance(self):
        lookup = run.masks(run.legacy.readout(2, 'weight_le_2'))
        fun = lambda t, x: run.features(t, x, 2, lookup)
        _, _, jac = run.distance_and_jacobian(np.arange(6)/3.,
                    np.array([[.1, .3], [-.4, .2]]), fun)
        np.testing.assert_allclose(jac, 0., atol=1e-13)

    def test_rank_threshold(self):
        self.assertEqual(run.numeric_rank(np.array([20., .1, 1e-14]), 1e-7, 1e-9), 2)


if __name__ == '__main__':
    unittest.main()
