"""Bounded exact checks of the new analytic statements, not a new census."""
import itertools
import json
import hashlib
from pathlib import Path
import sympy as s

root = Path(__file__).resolve().parent
def wt(p):
    return sum(x != 0 for x in p)
def anti(p, q):
    return sum(a != 0 and b != 0 and a != b for a, b in zip(p, q)) % 2
def allowed(p, readout):
    return all(not anti(p, q) or tuple(a ^ b for a, b in zip(p, q)) in readout
               for q in readout)

gauge = []
for n in range(1, 5):
    words = [p for p in itertools.product(range(4), repeat=n) if wt(p)]
    for k in range(1, n + 1):
        readout = {p for p in words if wt(p) <= k}
        got = {p for p in words if allowed(p, readout)}
        expected = set(words) if k == n else {p for p in words if wt(p) == 1}
        assert got == expected
        gauge.append(dict(n=n, k=k, generators=len(got)))
    readout = {p for p in words if wt(p) == 1 and 3 in p}
    got = {p for p in words if allowed(p, readout)}
    assert got == {p for p in words if all(x in (0, 3) for x in p)}
    gauge.append(dict(n=n, readout="local_Z", generators=len(got)))

theta, x = s.symbols("theta x", real=True)
ry = s.Matrix([[s.cos(theta/2), -s.sin(theta/2)],
               [s.sin(theta/2), s.cos(theta/2)]])
rz = s.diag(s.exp(-s.I*x/2), s.exp(s.I*x/2))
psi = ry * rz * s.Matrix([1, 1]) / s.sqrt(2)
z = (s.conjugate(psi.T) * s.diag(1, -1) * psi)[0]
assert s.trigsimp(s.expand_complex(z) + s.sin(theta)*s.cos(x)) == 0
families = []
for n in range(1, 9):
    angles = s.symbols("t0:%d" % n, real=True)
    for d in range(1, n + 1):
        beta = s.Matrix([sum((q+1)*s.sin(angles[q])**2 for q in range(n) if q % d == j)
                         for j in range(d)])
        jac = beta.jacobian(angles)
        witness = jac.subs({a:s.pi/4 for a in angles})
        assert witness[:, :d].det() == s.factorial(d) and witness.rank() == d
        assert all(any(jac[j,q] != 0 for j in range(d)) for q in range(n))
        families.append(dict(n=n, d=d, weights="w_q=q+1 (zero-based q)",
                             beta_minor=int(s.factorial(d)), rank=d))

a,b = s.symbols("a b", real=True)
beta = s.sin(a)**2 + s.sin(b)**2
loss = s.exp(-beta) + beta**2
grad = s.Matrix([s.diff(loss, a), s.diff(loss, b)])
null = s.Matrix([s.sin(2*b), -s.sin(2*a)])
assert s.simplify((grad.T*null)[0]) == 0
j = s.Matrix([[s.diff(beta,a),s.diff(beta,b)]])
assert j.subs({a:s.pi/4,b:0}) == s.Matrix([[1,0]])
assert j.subs({a:0,b:s.pi/4}) == s.Matrix([[0,1]])

# Endpoint Pauli transport through forward nearest-neighbor CNOTs.
def cx(p, control, target):
    # Binary x/z coordinates; phase is irrelevant for endpoint equality.
    xb=[int(t in (1,2)) for t in p]; zb=[int(t in (2,3)) for t in p]
    xb[target] ^= xb[control]; zb[control] ^= zb[target]
    return tuple({(0,0):0,(1,0):1,(1,1):2,(0,1):3}[x,z] for x,z in zip(xb,zb))
for n in range(2, 13):
    for p in [(3,)+(0,)*(n-1), (0,)*(n-1)+(1,)]:
        for q in range(n-1):
            assert cx(p,q,q+1) == p

result = dict(passed=True, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              sympy_version=s.__version__, gauge_checks=gauge, hub_grouped_checks=families,
              exact_single_qubit_feature=True, instantaneous_gradient_check=True,
              moving_gradient_subspace_counterexample=True,
              endpoint_checks_sizes=list(range(2,13)),
              kyriienko_all_size_equality="NOT CLAIMED",
              scope="Exact finite checks support but do not replace the written all-size proofs; no new source census or timing run.")
(root / "analysis/extension_review_checks.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"passed":True,"gauge_cases":len(gauge),"grouped_family_cases":len(families),
                  "endpoint_sizes":11,"new_census_settings":0},indent=2))
