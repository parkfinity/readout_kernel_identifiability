"""Independent small enumerations and numerical controls; run after freeze."""
import itertools
import unittest
import numpy as np
import run as exp


class Tests(unittest.TestCase):
    def test_fixed_metrics_and_target(self):
        for design in exp.designs():
            m = np.array(design["metric"])
            self.assertAlmostEqual(np.trace(m), 1)
            self.assertGreater(np.linalg.eigvalsh(m).min(), 0)
            self.assertEqual(len(design["sets"]), 4+design["products"])
        self.assertTrue(np.array_equal(exp.load_frozen()["target_basis"], np.eye(4)))

    def test_frozen_original_inputs(self):
        frozen = exp.load_frozen()
        self.assertEqual(len(frozen["points"]), 15)
        for point in frozen["points"]:
            self.assertEqual(len(point["x"]), 32)
            self.assertEqual(len(point["theta"]), 8)
            rng = np.random.default_rng(point["original_generation_seed"])
            for _ in range(point["point"]+1):
                theta = rng.uniform(-np.pi, np.pi, 8)
                x = rng.uniform(-.95, .95, (32, 2))
            np.testing.assert_array_equal(theta, point["theta"])
            np.testing.assert_array_equal(x, point["x"])

    def test_dense_source_and_tail(self):
        frozen = exp.load_frozen()
        for point in frozen["points"]:
            self.assertLess(max(exp.source_check(frozen, point).values()), 1e-8)

    def test_exhaustive_feature_covariance_two_shots(self):
        mu = np.array([[.3, -.2, .7, -.5]])
        pr = exp.probabilities(mu)[0]
        pairs = np.array(list(itertools.product(range(16), repeat=2)))
        weights = pr[pairs[:, 0]]*pr[pairs[:, 1]]
        counts = np.eye(16, dtype=int)[pairs].sum(axis=1)[None, :, None, :]
        for design in exp.designs():
            for estimator in exp.ESTIMATORS:
                draws = exp.sample_features(counts, design["sets"], 2, estimator)[0, :, 0, :]
                mean = weights @ draws
                centered = draws-mean
                cov = np.einsum("r,ri,rj->ij", weights, centered, centered)
                truth = np.array([np.prod(mu[0, s]) for s in design["sets"]])
                np.testing.assert_allclose(mean, truth, atol=1e-14)
                np.testing.assert_allclose(cov, exp.feature_covariance(mu, design["sets"], 2, estimator)[0], atol=1e-14)

    def test_conditional_expectation_two_shots(self):
        # For each vector of marginal counts, uniform permutations are independent.
        sequences = exp.BITS[np.array(list(itertools.product(range(16), repeat=2)))]
        local = sequences.mean(axis=1)
        for design in exp.designs():
            direct = np.stack([np.prod(sequences[:, :, s], axis=2).mean(axis=1) for s in design["sets"]], axis=1)
            for counts in np.unique(local, axis=0):
                group = np.all(local == counts, axis=1)
                expected = np.array([np.prod(counts[s]) for s in design["sets"]])
                np.testing.assert_allclose(direct[group].mean(axis=0), expected, atol=1e-14)

    def test_distance_variance_exhaustive_one_shot_halves(self):
        mu = np.array([[.3, -.2, .7, -.5], [-.4, .6, -.1, .8]])
        pr = exp.probabilities(mu)
        indices = np.array(list(itertools.product(range(16), repeat=2)))
        probabilities = pr[0, indices[:, 0]]*pr[1, indices[:, 1]]
        for design in exp.designs():
            m = np.array(design["metric"])
            signs = np.stack([np.prod(exp.BITS[:, s], axis=1) for s in design["sets"]], axis=1)
            difference = signs[indices[:, 0]]-signs[indices[:, 1]]
            distances = difference @ m @ difference.T
            joint = np.outer(probabilities, probabilities)
            empirical_mean = np.sum(joint*distances)
            empirical_variance = np.sum(joint*(distances-empirical_mean)**2)
            f = np.stack([np.prod(mu[:, s], axis=1) for s in design["sets"]], axis=1)
            delta = f[:1]-f[1:]
            cov = exp.feature_covariance(mu, design["sets"], 1, "direct_parity")
            self.assertAlmostEqual(empirical_mean, float(delta[0]@m@delta[0]), places=13)
            self.assertAlmostEqual(empirical_variance, exp.distance_variance(delta, cov, m)[0], places=13)

    def test_angle_jacobian(self):
        point = exp.load_frozen()["points"][0]
        theta, x = np.array(point["theta"]), point["x"]
        mu, dmu = exp.local_model(theta, x)
        for design in exp.designs():
            m = np.array(design["metric"])
            f, df = exp.features(mu, dmu, design["sets"])
            delta, ddelta = f[:16]-f[16:], df[:16]-df[16:]
            jac = 2*np.einsum("pi,ij,pjk->pk", delta, m, ddelta)
            numerical = []
            for q in range(4):
                d = np.eye(8)[q]*1e-5
                distances = []
                for shifted in [theta+d, theta-d]:
                    ff, _ = exp.features(*exp.local_model(shifted, x), design["sets"])
                    dd = ff[:16]-ff[16:]
                    distances.append(np.einsum("pi,ij,pj->p", dd, m, dd))
                numerical.append((distances[0]-distances[1])/(2e-5))
            np.testing.assert_allclose(jac, np.array(numerical).T, atol=1e-9, rtol=1e-7)

    def test_same_counts_and_shot_accounting(self):
        mu = np.array([[.3, -.2, .7, -.5], [-.4, .6, -.1, .8]])
        budget = 256
        counts = exp.shot_counts(exp.probabilities(mu), budget//2, 100, np.random.default_rng(19))
        np.testing.assert_array_equal(counts.sum(axis=3), budget//2)
        self.assertEqual(int(counts.sum()), 2*100*2*(budget//2))
        all_f = exp.sample_features(counts, exp.designs()[-1]["sets"], budget//2, "direct_parity")
        for design in exp.designs():
            direct = exp.sample_features(counts, design["sets"], budget//2, "direct_parity")
            product = exp.sample_features(counts, design["sets"], budget//2, "product_marginals")
            np.testing.assert_array_equal(direct[:, :, :, :4], all_f[:, :, :, :4])
            np.testing.assert_array_equal(direct[:, :, :, :4], product[:, :, :, :4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
