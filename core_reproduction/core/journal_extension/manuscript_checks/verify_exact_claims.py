"""Small exact review checks, NOT the 102-setting survey.

Python 3.8+; dependency: SymPy. No numeric rank threshold is used.
The representative normal form is the one in manuscript v5, Eq. (normal).
Prints a reproducible certificate to stdout; does not modify archived results.
"""
import hashlib
import json
from pathlib import Path
import platform
import sympy as s


def representative_certificate():
    beta = s.symbols('b0:4', real=True)
    p, pp, q, qp = s.symbols('p pp q qp', real=True)
    c, cp = beta[2]*p+beta[3], beta[2]*pp+beta[3]
    distance = (2*(s.cos(c)**2+s.cos(cp)**2)
                - 2*s.cos(c)*s.cos(cp)
                * (s.cos(beta[0]*(p-pp))+s.cos(beta[1]*(q-qp))))
    points = [
        (s.pi/6, s.pi/2, s.pi/6, 2*s.pi/3),
        (s.pi/6, s.pi/3, s.pi/3, s.pi/2),
        (s.pi/6, 2*s.pi/3, s.pi/6, 5*s.pi/6),
        (2*s.pi/3, s.pi/6, s.pi/6, s.pi/2),
    ]
    fixed_beta = dict(zip(beta, (1, 1, 1, 0)))
    grad = [s.diff(distance, b).subs(fixed_beta) for b in beta]
    jac = s.Matrix([[s.simplify(g.subs(dict(zip((p, pp, q, qp), pt))))
                     for g in grad] for pt in points])
    scaled = jac * s.diag(1/s.pi, 1/s.pi, 1/s.pi, 1)
    expected = s.Matrix([
        [0, 0, s.sqrt(3)/12, -s.sqrt(3)/2],
        [s.sqrt(3)/24, s.sqrt(3)/24, s.sqrt(3)/12, 0],
        [-s.sqrt(3)/4, -s.Rational(1, 2), -s.Rational(11, 24)+s.sqrt(3)/2, -s.Rational(1, 2)],
        [-s.sqrt(3)/4, -s.Rational(1, 4), s.Rational(11, 24)+s.sqrt(3)/2, s.Rational(1, 2)],
    ])
    assert (scaled-expected).applyfunc(s.simplify) == s.zeros(4)
    det_scaled = s.simplify(scaled.det())
    assert s.simplify(det_scaled-(2*s.sqrt(3)-39)/1536) == 0
    # This is a Q(sqrt(3)) identity; nonzero follows also from sqrt(3)<2.
    assert s.to_number_field(det_scaled, s.sqrt(3)).as_expr() != 0
    theta = s.symbols('t0:8', real=True)
    source_d = distance.subs(dict(zip(beta, (theta[0], theta[1]+theta[5], theta[4], theta[6]))))
    upper = [s.diff(source_d, theta[k]) for k in (2, 3, 7)]
    upper.append(s.diff(source_d, theta[1])-s.diff(source_d, theta[5]))
    assert all(s.simplify(v) == 0 for v in upper)
    # Exact feasibility: all lifted coordinates lie in (0,pi), x=cos(phi).
    assert all(0 < value/s.pi < 1 for pt in points for value in pt)
    return {
        'normal_form': 'RX0(a) RX1(b) RX0X1(c), initial |00>, local XYZ, unit weights',
        'beta_order': ['theta0', 'theta1+theta5', 'theta4', 'theta6'],
        'beta_star': [1, 1, 1, 0],
        'theta_star': [1, 1, 0, 0, 1, 0, 0, 0],
        'point_order': ['phi0(x)', 'phi0(xprime)', 'phi1(x)', 'phi1(xprime)'],
        'points': [[str(v) for v in pt] for pt in points],
        'scaled_jacobian': [[str(v) for v in row] for row in scaled.tolist()],
        'det_scaled': str(det_scaled),
        'det_jacobian': str(s.pi**3*det_scaled),
        'nonzero_proof': 'sqrt(3)<2, hence 2*sqrt(3)-39<0; pi>0',
        'null_basis': ['e2', 'e3', 'e7', 'e1-e5'],
        'generic_rank_lower': 4, 'generic_rank_upper': 4,
        'scope': 'smooth factorization dimension; not every-point rank, gate count, or shot cost',
    }


def counterexamples():
    a, ap, t = s.symbols('a ap t', real=True)
    lifted = (s.cos(a)-s.cos(ap))**2
    deriv = s.diff(lifted, a)+s.diff(lifted, ap)
    actual = lifted.subs({a: t, ap: t})
    witness = s.simplify(deriv.subs({a: 0, ap: s.pi/2}))
    assert actual == 0 and witness == 2
    x, xp = s.symbols('x xp', real=True)
    bias_d = (s.cos(x+t)-s.cos(xp+t))**2
    bias_witness = s.simplify(s.diff(bias_d, t).subs({x: s.pi/3, xp: s.pi/2, t: 0}))
    assert bias_witness == 1-s.sqrt(3)/2
    # The inverse-pair example is removed by ordinary adjacent fusion too.
    I = s.eye(2); X = s.Matrix([[0, 1], [1, 0]]); Y = s.Matrix([[0, -s.I], [s.I, 0]])
    def rot(P, v):
        return s.cos(v/2)*I-s.I*s.sin(v/2)*P
    b = s.symbols('b', real=True)
    pair = (rot(Y, -b)*rot(Y, b)).applyfunc(s.simplify)
    assert pair == I
    assert (rot(X, -a)*pair*rot(X, a)).applyfunc(s.simplify) == I
    return {'overlift': {'actual_d2': str(actual), 'lifted_derivative': str(deriv),
                        'infeasible_nonzero_witness': str(witness),
                        'conclusion': 'constant coefficients alone do not license lifted ACTIVE'},
            'common_angle_bias': {'nonzero_derivative': str(bias_witness)},
            'inverse_pair': 'exact identity, but also detected by adjacent same-axis fusion'}


def main():
    result = {'python': platform.python_version(), 'sympy': s.__version__,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'representative': representative_certificate(),
              'counterexamples': counterexamples(), 'passed': True,
              'survey_run': False}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
