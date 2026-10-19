"""Regression tests for exact single-product identifiability certificates."""

import itertools
import unittest

import sympy as sp

import verify_one_product as v


class OneProductVerificationTests(unittest.TestCase):
    def test_fixed_metric_is_spd_and_has_trace_seven(self):
        self.assertEqual(v.METRIC, v.METRIC.T)
        self.assertEqual(v.METRIC.eigenvals(), {sp.Integer(1): 4, sp.Integer(3): 1})
        self.assertEqual([v.METRIC[:i, :i].det() for i in range(1, 6)], [2, 3, 3, 3, 3])
        self.assertEqual(sp.trace(v.METRIC), 7)

    def test_augmented_features_have_the_correct_signs(self):
        f = v.COEFFICIENTS * sp.Matrix([v.U, v.V, v.U**2])
        self.assertEqual(f, sp.Matrix([-v.A*v.U, -v.B*v.V, -v.C*v.U,
                                       -v.D*v.V, v.A*v.C*v.U**2]))
        self.assertEqual(f[4], f[0] * f[2])

    def test_coefficient_gram(self):
        self.assertEqual(v.GRAM, sp.Matrix([
            [2*v.A**2+v.C**2, v.A*v.B, 0],
            [v.A*v.B, 2*v.B**2+v.D**2, 0], [0, 0, v.A**2*v.C**2]]))

    def test_exact_amplitude_determinant(self):
        expected = -8 * v.A**2 * v.C * v.D * (2*v.A**2-v.C**2)
        self.assertEqual(sp.expand(v.DETERMINANT - expected), 0)

    def test_angular_chain_rule_by_direct_differentiation(self):
        chain_rule = v.JACOBIAN.subs(v.amplitude_substitution()) * sp.diag(*map(sp.cos, v.THETA))
        self.assertEqual(v.angular_jacobian(), chain_rule)

    def test_angular_witness(self):
        point = dict.fromkeys(v.THETA, sp.pi / 6)
        self.assertEqual(v.angular_jacobian().subs(point), v.WITNESS)
        self.assertEqual(sp.simplify(v.WITNESS.det()), sp.Rational(-9, 128))
        self.assertEqual(v.WITNESS.rank(), 4)

    def test_unit_trace_scaling(self):
        normalized_gram = v.COEFFICIENTS.T * (v.METRIC / 7) * v.COEFFICIENTS
        self.assertEqual(normalized_gram, v.GRAM / 7)
        normalized_jacobian = (v.COORDINATES / 7).jacobian(v.AMPLITUDES)
        self.assertEqual(sp.factor(normalized_jacobian.det()), v.DETERMINANT / 7**4)
        self.assertEqual(sp.trace(v.METRIC / 7), 1)
        self.assertEqual(sp.simplify((v.WITNESS / 7).det()), sp.Rational(-9, 307328))

    def test_actual_domain_affine_basis_independence(self):
        evaluation = v.basis_evaluation_matrix()
        self.assertNotEqual(sp.simplify(evaluation.det()), 0)
        self.assertEqual(evaluation.rank(), 4)

    def test_actual_input_pairs_detect_all_four_gram_coordinates(self):
        design = v.distance_design_matrix()
        self.assertNotEqual(sp.simplify(design.det()), 0)
        self.assertEqual(design.rank(), 4)
        self.assertEqual((design * v.WITNESS).rank(), 4)

    def test_fourier_conversion_separates_constant_offset(self):
        fourier_c = v.COEFFICIENTS.copy()
        fourier_c[4, 2] = v.A * v.C / 2
        offset = sp.Matrix([0, 0, 0, 0, v.A*v.C/2])
        h_fourier = sp.Matrix([v.U, v.V, 2*v.U**2-1])
        difference = fourier_c*h_fourier + offset - v.COEFFICIENTS*sp.Matrix([v.U, v.V, v.U**2])
        self.assertEqual(difference.applyfunc(sp.expand), sp.zeros(5, 1))
        fourier_gram = fourier_c.T * v.METRIC * fourier_c
        coordinates = sp.Matrix([fourier_gram[0, 0], fourier_gram[0, 1],
                                 fourier_gram[1, 1], fourier_gram[2, 2]])
        self.assertEqual(sp.factor(coordinates.jacobian(v.AMPLITUDES).det()), v.DETERMINANT / 4)

    def test_original_fixed_metric_has_at_most_three_coordinates(self):
        entries = sp.symbols("m0:10", real=True)
        metric = sp.zeros(4)
        for entry, (i, j) in zip(entries, itertools.combinations_with_replacement(range(4), 2)):
            metric[i, j] = metric[j, i] = entry
        gram = v.LOCAL_COEFFICIENTS.T * metric * v.LOCAL_COEFFICIENTS
        coordinates = sp.Matrix([gram[0, 0], gram[0, 1], gram[1, 1]])
        self.assertEqual(coordinates.jacobian(v.AMPLITUDES).shape, (3, 4))
        self.assertEqual(gram.shape, (2, 2))

    def test_every_single_pairwise_product_diagonal_ceiling(self):
        for i, j in itertools.combinations_with_replacement(range(4), 2):
            with self.subTest(pair=(i, j)):
                coordinates = v.one_product_diagonal_coordinates(i, j)
                jacobian = coordinates.jacobian(v.AMPLITUDES)
                self.assertEqual(jacobian.shape, (3, 4))
                self.assertEqual(jacobian.rank(), 3)

    def test_arbitrary_scalar_diagonal_ceiling_would_be_false(self):
        w, coordinates = v.sum_product_diagonal_coordinates()
        expected = -16*v.A*v.B*v.C*v.D*w[4]**2 * (w[0]*v.A**2-w[2]*v.C**2) * (w[1]*v.B**2-w[3]*v.D**2)
        actual = sp.factor(coordinates.jacobian(v.AMPLITUDES).det())
        self.assertEqual(sp.expand(actual - expected), 0)
        self.assertNotEqual(actual, 0)

    def test_generic_is_not_everywhere(self):
        self.assertEqual(v.angular_jacobian().subs(dict.fromkeys(v.THETA, 0)).rank(), 0)
        singular = dict(zip(v.AMPLITUDES, [sp.Rational(1, 2), sp.Rational(1, 2),
                                                         sp.sqrt(2)/2, sp.Rational(1, 2)]))
        self.assertEqual(sp.simplify(v.DETERMINANT.subs(singular)), 0)
        self.assertEqual(v.JACOBIAN.subs(singular).rank(), 3)

    def test_global_nonuniqueness_at_positive_regular_points(self):
        first, second = v.positive_nonunique_points()
        self.assertNotEqual(first, second)
        self.assertEqual(v.COORDINATES.subs(first), v.COORDINATES.subs(second))
        self.assertEqual(v.COORDINATES.subs(first), sp.Matrix([sp.Rational(3, 4), sp.Rational(1, 8),
                                                            sp.Rational(3, 8), sp.Rational(1, 16)]))
        for point in (first, second):
            self.assertNotEqual(sp.simplify(v.DETERMINANT.subs(point)), 0)
            self.assertTrue(all(0 < value < 1 for value in point.values()))

    def test_product_state_z_parity_identity(self):
        means, total, locals_, parity = v.z_moments()
        self.assertEqual(total, 1)
        self.assertEqual(locals_, list(means))
        self.assertEqual(parity, means[0]*means[2])
        self.assertEqual(parity.subs({means[0]: -v.A*v.U, means[2]: -v.C*v.U}), v.A*v.C*v.U**2)

    def test_finite_shot_estimators_are_not_identical(self):
        z0 = (1, -1)
        z2 = (1, -1)
        direct_parity = sp.Rational(sum(a*b for a, b in zip(z0, z2)), 2)
        product_of_means = sp.Rational(sum(z0), 2) * sp.Rational(sum(z2), 2)
        self.assertEqual(direct_parity, 1)
        self.assertEqual(product_of_means, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
