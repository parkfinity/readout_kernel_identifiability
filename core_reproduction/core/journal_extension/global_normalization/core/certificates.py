"""Sufficient ideal projected-kernel certificates; UNKNOWN is not ACTIVE.

Python >=3.8, NumPy only. No claim of algorithmic novelty or general equivalence.
Never use fidelity suffix removal on a projected kernel.
"""
from dataclasses import asdict
import numpy as np
from structural_core import commute, conjugate_pauli, states_and_tangent, apply_pauli


def validate_readout(observables, weights, n):
    if not observables or len(set(observables)) != len(observables):
        raise ValueError('Readout must be nonempty and contain distinct Pauli words')
    if any(len(p) != n or any(c not in 'IXYZ' for c in p) for p in observables):
        raise ValueError('Invalid Pauli readout')
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(observables),) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Finite positive diagonal weights required')
    return w


def product_word(p, q):
    """Pauli product up to sign/phase (used only for closure)."""
    return ''.join(b if a == 'I' else a if b == 'I' else 'I' if a == b
                   else ({'X', 'Y', 'Z'} - {a, b}).pop() for a, b in zip(p, q))


def invariant_generator(p, observables, weights, rule='metric'):
    w = validate_readout(observables, weights, len(p))
    if any(a not in 'IXYZ' for a in p):
        raise ValueError('Invalid generator')
    if rule not in ('metric', 'commutant'):
        raise ValueError(rule)
    lookup = {o: j for j, o in enumerate(observables)}
    pairs = []
    for j, o in enumerate(observables):
        if commute(p, o):
            continue
        if rule == 'commutant':
            return False, {'reason': 'noncommuting_observable', 'observable': o}
        partner = product_word(p, o)
        if partner not in lookup:
            return False, {'reason': 'readout_not_closed', 'observable': o, 'partner': partner}
        # Exact equality, not a numerical tolerance, defines the certificate.
        if w[j] != w[lookup[partner]]:
            return False, {'reason': 'unequal_pair_weights', 'observable': o, 'partner': partner}
        pairs.append([o, partner])
    return True, {'rule': rule, 'rotation_pairs': pairs}


def reduce_circuit(circuit, observables, weights, method='transport_metric'):
    if method not in ('terminal_commutant', 'transport_commutant', 'terminal_metric', 'transport_metric'):
        raise ValueError(method)
    n = len(observables[0]) if observables else 0
    validate_readout(observables, weights, n)
    for g in circuit:
        if g.op == 'R':
            if not isinstance(g.scale, (int, float, np.integer, np.floating)) or not np.isfinite(g.scale):
                raise ValueError('Finite real constant angle scale required')
            if len(g.pauli) != n or any(c not in 'IXYZ' for c in g.pauli) or g.kind not in ('fixed','param','product','data'):
                raise ValueError('Unsupported/invalid rotation')
        elif g.op not in ('H', 'CX'):
            raise ValueError('Unsupported gate')
        elif len(g.wires) != (1 if g.op == 'H' else 2) or len(set(g.wires)) != len(g.wires) or any(q < 0 or q >= n for q in g.wires):
            raise ValueError('Invalid Clifford wires')
    work = list(enumerate(circuit))
    certs = []
    transport = method.startswith('transport_')
    rule = method.split('_')[1]
    changed = True
    while changed:
        changed = False
        for i in range(len(work) - 1, -1, -1):
            original, gate = work[i]
            if gate.op != 'R' or gate.kind != 'param':
                continue
            if not transport and i != len(work) - 1:
                continue
            p, steps, safe = gate.pauli, [], True
            for later_index, later in work[i + 1:]:
                if later.op == 'R' and commute(p, later.pauli):
                    steps.append({'index': later_index, 'rule': 'commute'})
                elif later.op in ('H', 'CX'):
                    p = conjugate_pauli(p, later)
                    steps.append({'index': later_index, 'rule': 'clifford', 'generator_up_to_sign': p})
                else:
                    safe = False
                    break
            if not safe:
                continue
            ok, detail = invariant_generator(p, observables, weights, rule)
            if ok:
                certs.append({'original_index': original, 'gate': asdict(gate),
                              'output_generator_up_to_sign': p, 'steps': steps, 'readout_proof': detail})
                work.pop(i)
                changed = True
                break
    reduced = tuple(g for _, g in work)
    original_ids = {g.parameter for g in circuit if g.parameter is not None}
    retained_ids = {g.parameter for g in reduced if g.parameter is not None}
    return reduced, {'method': method, 'certificates': certs,
                     'inactive_parameters': sorted(original_ids - retained_ids),
                     'unknown_parameters': sorted(retained_ids)}


def projected_kernel(circuit, X, theta, n, observables, weights, kind='rbf', gamma=0.5, parameter=None):
    w = validate_readout(observables, weights, n)
    if kind not in ('rbf', 'dot') or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('Invalid kernel')
    s, t = states_and_tangent(circuit, X, theta, n, parameter)
    features, tangents = [], []
    for o in observables:
        ps = apply_pauli(s, o)
        features.append(np.real(np.sum(s.conj() * ps, axis=1)))
        tangents.append(2 * np.real(np.sum(t.conj() * ps, axis=1)))
    f, df = np.column_stack(features), np.column_stack(tangents)
    if kind == 'dot':
        return (f * w) @ f.T, (df * w) @ f.T + (f * w) @ df.T, f
    delta, ddelta = f[:, None, :] - f[None, :, :], df[:, None, :] - df[None, :, :]
    distance = np.sum(w * delta ** 2, axis=-1)
    K = np.exp(-gamma * distance)
    dK = -2 * gamma * K * np.sum(w * delta * ddelta, axis=-1)
    return K, dK, f


def rbf_output_derivative_bound(generator, observables, weights):
    """Uniform |d k/d theta| <= max_pairs |wi-wj|/sqrt(wi*wj)/e.

    Only common OUTPUT exp(-i theta P/2), closed readout, positive weights.
    This is an elementary norm bound, not a new-theorem claim. No product angles.
    """
    w = validate_readout(observables, weights, len(generator))
    lookup = {o: j for j, o in enumerate(observables)}
    defect = 0.0
    for i, o in enumerate(observables):
        if not commute(generator, o):
            partner = product_word(generator, o)
            if partner not in lookup:
                raise ValueError('Bound requires closed readout')
            j = lookup[partner]
            defect = max(defect, abs(w[i] - w[j]) / np.sqrt(w[i] * w[j]))
    return float(defect / np.e)
