"""
Numerical sanity checks for the generic kernel differential rank r of the
one-layer Kyriienko-type family (chebyshev_tower encoding, RZ-RX-RZ local block,
HEA even-then-odd CNOT entangler), as defined in
  cross_family_resource_v1/core/vendor/.../kyriienko_nonlinear_encoding_circuit.py
and frozen in certified_dimension_extension_v1/frozen_inputs/.

We compute r = grank D_theta vech G(theta) via the equivalent sampled statistic
  r = rank of the Jacobian of  V(theta) = [ d2_theta(x_p, x'_p) ]_p
where d2_theta(x,x') = sum_i w_i ( f_i(x) - f_i(x') )^2  (readout metric M=diag(w)),
f_i(x) = <0| D(x)^dag L(theta)^dag E^dag O_i E L(theta) D(x) |0>.

Gradients use the EXACT parameter-shift rule for Pauli-generated rotations, so the
reported ranks are not corrupted by finite-difference error (singular values are
either O(1) or ~1e-14).  Labelled NUMERICAL throughout.

All claims about r here are numerical; the manuscript certifies n=2,4 exactly.
"""
import numpy as np
import itertools

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}


def kron_list(mats):
    out = np.array([[1]], dtype=complex)
    for m in mats:
        out = np.kron(out, m)
    return out


def pauli_string_op(s):
    return kron_list([PAULI[c] for c in s])


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t):
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]], dtype=complex)


def rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def apply_1q(psi, U1, q, n):
    psi = psi.reshape([2] * n)
    psi = np.tensordot(U1, psi, axes=([1], [q]))
    psi = np.moveaxis(psi, 0, q)
    return psi.reshape(-1)


def apply_cx(psi, ctrl, tgt, n):
    psi = psi.reshape([2] * n)
    # flip tgt where ctrl==1
    idx = [slice(None)] * n
    idx[ctrl] = 1
    sub = psi[tuple(idx)]
    sub = np.flip(sub, axis=(tgt if tgt < ctrl else tgt - 1))
    psi[tuple(idx)] = sub
    return psi.reshape(-1)


def encoding_angles(n):
    """Return list of (qubit, feature_index, angle_coefficient) R_y gates for
    chebyshev_tower, exactly as in KyriienkoEncodingCircuit (alpha=2, ry).
    angle = coeff * arccos(x_feature).  num_features=2."""
    gates = []
    alpha = 2.0
    num_features = 2
    iqubit = 0
    index_offset_encoding = 0
    icheb = 1
    for outer_ in range(n):           # num_chebyshev = n
        for inner_ in range(num_features):
            f = index_offset_encoding % num_features
            q = iqubit % n
            coeff = alpha * icheb      # multiplies arccos(x_f)
            gates.append((q, f, coeff))
            iqubit += 1
            index_offset_encoding += 1
        icheb += 1                     # tower=True
    return gates


def build_state(n, x, theta):
    """psi = E L(theta) D(x) |0...0>.  x = (x0,x1), theta length 3n."""
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    # D(x): encoding R_y gates in order
    for (q, f, coeff) in encoding_angles(n):
        ang = coeff * np.arccos(x[f])
        psi = apply_1q(psi, ry(ang), q, n)
    # L(theta): RZ RX RZ per qubit
    off = 0
    for q in range(n):
        psi = apply_1q(psi, rz(theta[off]), q, n)
        psi = apply_1q(psi, rx(theta[off + 1]), q, n)
        psi = apply_1q(psi, rz(theta[off + 2]), q, n)
        off += 3
    # E: HEA even pairs then odd pairs
    for start in (0, 1):
        for i in range(start, n - 1, 2):
            psi = apply_cx(psi, i, i + 1, n)
    return psi


def readout_paulis(n, kind):
    """Return list of Pauli strings (len n) for the readout set."""
    if kind == 'local_xyz':
        ops = []
        for q in range(n):
            for P in 'XYZ':
                s = ['I'] * n
                s[q] = P
                ops.append(''.join(s))
        return ops
    if kind.startswith('weight_le_'):
        k = int(kind.split('_')[-1])
        ops = []
        for w in range(1, k + 1):
            for sites in itertools.combinations(range(n), w):
                for assign in itertools.product('XYZ', repeat=w):
                    s = ['I'] * n
                    for site, P in zip(sites, assign):
                        s[site] = P
                    ops.append(''.join(s))
        return ops
    raise ValueError(kind)


def expectations(psi, pauli_ops_cached):
    """<psi|O|psi> for each cached (dense) operator; real part."""
    out = np.empty(len(pauli_ops_cached))
    for idx, O in enumerate(pauli_ops_cached):
        out[idx] = np.real(np.vdot(psi, O @ psi))
    return out


def feature_fn(n, theta, Xsamples, Ocache):
    """Return array [n_samples, n_obs] of features f_i(x)."""
    F = np.empty((len(Xsamples), len(Ocache)))
    for p, x in enumerate(Xsamples):
        psi = build_state(n, x, theta)
        F[p] = expectations(psi, Ocache)
    return F


def kernel_rank(n, kind, n_pairs=400, seed=0, tol=1e-8, theta=None):
    rng = np.random.default_rng(seed)
    ops = readout_paulis(n, kind)
    Ocache = [pauli_string_op(s) for s in ops]
    m = len(ops)
    K = 3 * n
    if theta is None:
        theta = rng.uniform(-2 * np.pi, 2 * np.pi, size=K)
    # sample input pairs in (-1,1)^2
    xs = rng.uniform(-0.98, 0.98, size=(n_pairs, 2))
    xps = rng.uniform(-0.98, 0.98, size=(n_pairs, 2))
    allX = list(xs) + list(xps)

    def d2_vector(th):
        F = feature_fn(n, th, allX, Ocache)
        Fx = F[:n_pairs]
        Fxp = F[n_pairs:]
        diff = Fx - Fxp               # [n_pairs, m]
        return np.sum(diff * diff, axis=1)   # [n_pairs]

    # exact parameter-shift Jacobian of f, then chain rule to d2
    Fx0 = feature_fn(n, theta, allX, Ocache)
    Fx = Fx0[:n_pairs]
    Fxp = Fx0[n_pairs:]
    diff0 = Fx - Fxp
    J = np.empty((n_pairs, K))
    sh = np.pi / 2
    for j in range(K):
        tp = theta.copy(); tp[j] += sh
        tm = theta.copy(); tm[j] -= sh
        Fp = feature_fn(n, tp, allX, Ocache)
        Fm = feature_fn(n, tm, allX, Ocache)
        dF = 0.5 * (Fp - Fm)          # exact d f / d theta_j  [2*n_pairs, m]
        dFx = dF[:n_pairs]; dFxp = dF[n_pairs:]
        ddiff = dFx - dFxp
        J[:, j] = 2.0 * np.sum(diff0 * ddiff, axis=1)
    sv = np.linalg.svd(J, compute_uv=False)
    smax = sv[0] if sv.size else 0.0
    rank = int(np.sum(sv > tol * max(smax, 1e-30)))
    return rank, sv, K, m


if __name__ == '__main__':
    import sys
    print("n, readout, m(obs), K(=3n), r_numeric, 3n-2, match, sv_gap")
    results = []
    for n in range(2, 9):
        kinds = ['local_xyz']
        for k in range(2, n):
            kinds.append(f'weight_le_{k}')
        kinds.append(f'weight_le_{n}')   # k=n -> expect r=0
        for kind in kinds:
            # keep it feasible: skip very large observable sets
            try:
                rank, sv, K, m = kernel_rank(n, kind, n_pairs=max(200, 20 * n), seed=1)
            except MemoryError:
                print(f"{n}, {kind}: SKIP (memory)")
                continue
            target = 3 * n - 2
            if kind == f'weight_le_{n}':
                target = 0
            # singular value gap diagnostic
            nz = sv[sv > 1e-8 * max(sv[0], 1e-30)]
            gap = (nz[-1] / sv[len(nz)]) if len(nz) < len(sv) and len(nz) > 0 else float('inf')
            match = 'OK' if rank == target else 'DIFF'
            print(f"{n}, {kind}, {m}, {K}, r={rank}, target={target}, {match}, gap~{gap:.1e}")
            results.append((n, kind, m, K, rank, target, match))
    print("\nSUMMARY mismatches:", [r for r in results if r[6] != 'OK'])
