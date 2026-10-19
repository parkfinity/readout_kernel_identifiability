#!/usr/bin/env python3
"""Exact, independent verification of the one-product recovery construction.

Requires Python 3 and SymPy. No numerical rank tolerances are used.
Run this file to print the certificate, optionally saving generated JSON with
``--output PATH``. Run ``python3 -m unittest discover -s verification_math -v``
from the revision directory to execute the accompanying regression tests.

The coordinate order is (G_uu, G_uv, G_vv, G_u2u2), and parameter order is
(a,b,c,d). Changing either order can change the determinant sign.
"""

import argparse
import itertools
import json
import platform
from pathlib import Path

import sympy as sp


A, B, C, D = sp.symbols("a b c d", real=True)
AMPLITUDES = (A, B, C, D)
U, V = sp.symbols("u v", real=True)
THETA = sp.symbols("theta0:4", real=True)
LOCAL_COEFFICIENTS = sp.Matrix([[-A, 0], [0, -B], [-C, 0], [0, -D]])
COEFFICIENTS = sp.Matrix(
    [[-A, 0, 0], [0, -B, 0], [-C, 0, 0], [0, -D, 0], [0, 0, A * C]]
)
METRIC = sp.Matrix(
    [[2, 1, 0, 0, 0], [1, 2, 0, 0, 0], [0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
)
GRAM = COEFFICIENTS.T * METRIC * COEFFICIENTS
COORDINATES = sp.Matrix([GRAM[0, 0], GRAM[0, 1], GRAM[1, 1], GRAM[2, 2]])
JACOBIAN = COORDINATES.jacobian(AMPLITUDES)
DETERMINANT = sp.factor(JACOBIAN.det())
HALF_POINT = dict.fromkeys(AMPLITUDES, sp.Rational(1, 2))
WITNESS = JACOBIAN.subs(HALF_POINT) * sp.diag(*([sp.sqrt(3) / 2] * 4))


def amplitude_substitution():
    return dict(zip(AMPLITUDES, map(sp.sin, THETA)))


def angular_jacobian():
    """Independent differentiation of sin-parameterized Gram entries."""
    return COORDINATES.subs(amplitude_substitution()).jacobian(THETA)


def basis_evaluation_matrix():
    """Rows evaluate (1,u,v,u^2) at actual-domain x in (-1,1)^2.

    The four input points are (0,0), (pi/6,0), (pi/4,0), (0,pi/6).
    Their entries are exact algebraic numbers; pi/4 < 1.
    """
    s = sp.sqrt(3) / 2
    t = sp.sqrt(2) / 2
    points = ((1, 1), (s, 1), (t, 1), (1, s))
    return sp.Matrix([[1, u, v, u * u] for u, v in points])


def distance_design_matrix():
    """Four actual-domain pairs detect the four Gram coordinates exactly.

    Each first input is (0,0). Second inputs are (pi/6,0), (pi/4,0),
    (0,pi/6), (pi/6,pi/6). Columns multiply (Guu,Guv,Gvv,Gu2u2).
    """
    s = sp.sqrt(3) / 2
    t = sp.sqrt(2) / 2
    points = ((s, 1), (t, 1), (1, s), (s, s))
    return sp.Matrix(
        [[(1 - u) ** 2, 2 * (1 - u) * (1 - v), (1 - v) ** 2,
          (1 - u * u) ** 2] for u, v in points]
    )


def positive_nonunique_points():
    """Two distinct regular points, with all amplitudes strictly in (0,1)."""
    return (
        {A: sp.Rational(1, 2), B: sp.Rational(1, 4),
         C: sp.Rational(1, 2), D: sp.Rational(1, 2)},
        {A: sp.sqrt(2) / 4, B: sp.sqrt(2) / 4,
         C: sp.sqrt(2) / 2, D: sp.sqrt(2) / 4},
    )


def one_product_diagonal_coordinates(i, j):
    """Every single pairwise product has only three diagonal Gram entries.

    Includes i=j (squares), although a physical two-qubit ZZ product has i<j.
    Its one added data mode is u^2, uv, or v^2.
    """
    weights = sp.symbols("w0:5", positive=True)
    coefficient = AMPLITUDES[i] * AMPLITUDES[j]
    return sp.Matrix([
        weights[0] * A**2 + weights[2] * C**2,
        weights[1] * B**2 + weights[3] * D**2,
        weights[4] * coefficient**2,
    ])


def sum_product_diagonal_coordinates():
    """Counterexample to extending the diagonal ceiling to any scalar lift.

    One scalar g=f0*f2+f1*f3 has two data modes, ac*u^2+bd*v^2.
    Four Gram entries already prove rank four with diagonal weights.
    """
    w = sp.symbols("w0:5", positive=True)
    coordinates = sp.Matrix([
        w[0] * A**2 + w[2] * C**2, w[4] * A**2 * C**2,
        w[1] * B**2 + w[3] * D**2, w[4] * B**2 * D**2,
    ])
    return w, coordinates


def z_moments():
    """Exact local and parity means for the product Z-outcome distribution.

    A diagonal tail preserves this distribution, even when it entangles the
    state. The calculation therefore applies to the stated source circuit.
    """
    means = sp.symbols("m0:4", real=True)
    local = [sp.Integer(0)] * 4
    pair = sp.Integer(0)
    total = sp.Integer(0)
    for signs in itertools.product((-1, 1), repeat=4):
        probability = sp.prod((1 + z * m) / 2 for z, m in zip(signs, means))
        total += probability
        pair += signs[0] * signs[2] * probability
        for q in range(4):
            local[q] += signs[q] * probability
    return means, sp.expand(total), [sp.expand(x) for x in local], sp.expand(pair)


def certificate():
    """Return only exact values and carefully qualified conclusions."""
    first, second = positive_nonunique_points()
    w, sum_coordinates = sum_product_diagonal_coordinates()
    sum_det = sp.factor(sum_coordinates.jacobian(AMPLITUDES).det())
    basis_det = sp.simplify(basis_evaluation_matrix().det())
    distance_det = sp.factor(distance_design_matrix().det())
    return {
        "python_version": platform.python_version(),
        "sympy_version": sp.__version__,
        "arithmetic": "Exact symbolic, rational and algebraic; no floating-point rank tests",
        "data_basis": ["u=cos(x0)", "v=cos(x1)", "u^2"],
        "parameter_order": [str(x) for x in AMPLITUDES],
        "gram_coordinate_order": ["G_uu", "G_uv", "G_vv", "G_u2u2"],
        "metric": [list(map(str, row)) for row in METRIC.tolist()],
        "metric_eigenvalues_with_multiplicity": {
            str(k): int(v) for k, v in METRIC.eigenvals().items()
        },
        "metric_leading_principal_minors": [str(METRIC[:n, :n].det()) for n in range(1, 6)],
        "metric_trace": str(sp.trace(METRIC)),
        "metric_condition_number": "3",
        "gram": [list(map(str, row)) for row in GRAM.tolist()],
        "amplitude_jacobian": [list(map(str, row)) for row in JACOBIAN.tolist()],
        "amplitude_determinant": str(DETERMINANT),
        "angular_determinant": str(
            DETERMINANT.subs(amplitude_substitution()) * sp.prod(map(sp.cos, THETA))
        ),
        "angular_witness_theta": ["pi/6"] * 4,
        "angular_witness_determinant": str(sp.simplify(WITNESS.det())),
        "unit_trace_metric_scale": "1/7",
        "unit_trace_angular_witness_determinant": str(sp.simplify(WITNESS.det() / 7**4)),
        "pure_fourier_basis": ["cos(x0)", "cos(x1)", "cos(2*x0)"],
        "pure_fourier_angular_witness_determinant": str(sp.simplify(WITNESS.det() / 4)),
        "pure_fourier_unit_trace_witness_determinant": str(sp.simplify(WITNESS.det() / (4 * 7**4))),
        "actual_domain": "(-1,1)^2; more generally the (u,v) image contains an open set",
        "affine_basis_witness_determinant": str(basis_det),
        "distance_design_witness_determinant": str(distance_det),
        "finite_distance_angular_witness_determinant": str(sp.factor(distance_det * WITNESS.det())),
        "generic_rank": 4,
        "global_injectivity": False,
        "positive_regular_nonunique_amplitudes": [
            [str(p[x]) for x in AMPLITUDES] for p in (first, second)
        ],
        "common_gram_coordinates_at_nonunique_points": [str(x) for x in COORDINATES.subs(first)],
        "amplitude_determinants_at_nonunique_points": [str(DETERMINANT.subs(p)) for p in (first, second)],
        "restricted_minimum_appended_scalar_count": 1,
        "minimality_scope": (
            "Keep the original four local means and use a fixed input- and "
            "parameter-independent SPD quadratic metric; zero appended outputs "
            "have rank <=3 and the specified single product attains generic rank4."
        ),
        "one_pairwise_product_with_fixed_positive_diagonal_metric_rank_ceiling": 3,
        "diagonal_ceiling_scope_warning": (
            "This is not a ceiling for an arbitrary single scalar feature. "
            "g=f0*f2+f1*f3 has generic rank4 under positive diagonal weighting."
        ),
        "sum_product_diagonal_counterexample_determinant": str(sum_det),
        "measurement_information": (
            "f0*f2 equals <Z0 Z2> for the product-pre-tail/diagonal-tail family; "
            "it is determined by exact local means. Direct parity and products "
            "of finite-shot sample means are distinct estimators."
        ),
        "scope": (
            "Generic differential identifiability of four retained angles only; "
            "not every parameter point, global uniqueness, tail-angle recovery, "
            "shot efficiency, statistical conditioning, or a universal feature-design optimum."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save the generated exact certificate as JSON")
    args = parser.parse_args()
    result = json.dumps(certificate(), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")


if __name__ == "__main__":
    main()
