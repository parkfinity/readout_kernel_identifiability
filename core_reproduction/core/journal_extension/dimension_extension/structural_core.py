"""Conservative ideal-fidelity certificates. NumPy only; Python >=3.8.

Circuit tuples are in application order. This is a research prototype, not a
claim of a new equivalence algorithm. Commutation/Clifford rules are known.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class Gate:
    op: str
    pauli: str = ""
    kind: str = "fixed"  # fixed, data, param, product
    index: int = 0
    feature: int = 0
    value: float = 0.0
    wires: tuple = ()
    scale: float = 1.0  # Constant angle multiplier; preserves tied CRZ parameters.

    @property
    def data_dependent(self):
        return self.op == "R" and self.kind in ("data", "product")

    @property
    def parameter(self):
        return self.index if self.op == "R" and self.kind in ("param", "product") else None


def pauli(n, axis, wire):
    return "I" * wire + axis + "I" * (n - wire - 1)


def rotation(n, axis, wire, kind, index=0, feature=0, value=0.0):
    return Gate("R", pauli(n, axis, wire), kind, index, feature, value)


def h(wire):
    return Gate("H", wires=(wire,))


def cx(control, target):
    return Gate("CX", wires=(control, target))


def commute(p, q):
    return sum(a != b and a != "I" and b != "I" for a, b in zip(p, q)) % 2 == 0


def conjugate_pauli(p, gate):
    """Return C P C^dagger up to sign, sufficient for commutation/deletion.

    We never use the sign-free result to evaluate a transformed circuit.
    Only the transported, input-independent output rotation is discarded.
    """
    if gate.op == "H":
        chars = list(p)
        q = gate.wires[0]
        chars[q] = {"X": "Z", "Z": "X", "Y": "Y", "I": "I"}[chars[q]]
        return "".join(chars)
    if gate.op == "CX":
        xs = [a in "XY" for a in p]
        zs = [a in "ZY" for a in p]
        c, t = gate.wires
        xs[t] ^= xs[c]
        zs[c] ^= zs[t]
        return "".join({(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}[(int(x), int(z))] for x, z in zip(xs, zs))
    raise ValueError("Unsupported Clifford: " + gate.op)


def canonicalize(circuit, method):
    if method in ("raw", "syntax"):
        return tuple(circuit), []
    if method not in ("suffix", "commute"):
        raise ValueError(method)
    work = list(enumerate(circuit))
    certificates = []

    def strip():
        while work and not work[-1][1].data_dependent:
            original_index, gate = work.pop()
            certificates.append({"original_index": original_index, "rule": "terminal", "gate": asdict(gate)})

    strip()
    if method == "commute":
        changed = True
        while changed:
            changed = False
            for i in range(len(work) - 1, -1, -1):
                original_index, gate = work[i]
                if gate.op != "R" or gate.data_dependent:
                    continue
                transported = gate.pauli
                steps = []
                safe = True
                for later_index, later in work[i + 1:]:
                    if later.op == "R" and commute(transported, later.pauli):
                        steps.append({"across_original_index": later_index, "rule": "commuting_paulis"})
                    elif later.op in ("H", "CX"):
                        transported = conjugate_pauli(transported, later)
                        steps.append({"across_original_index": later_index, "rule": "clifford_conjugation_up_to_sign", "pauli": transported})
                    else:
                        safe = False
                        break
                if safe:
                    certificates.append({"original_index": original_index, "rule": "transport_to_output", "gate": asdict(gate), "steps": steps})
                    work.pop(i)
                    strip()
                    changed = True
                    break
    return tuple(g for _, g in work), certificates


def signature(circuit, theta):
    # Values for all retained parameters are included. Distinct initializations
    # must never collide merely because the circuit syntax is identical.
    ids = sorted({g.parameter for g in circuit if g.parameter is not None})
    payload = {"gates": [asdict(g) for g in circuit], "retained_bindings": [(p, float(theta[p]).hex()) for p in ids]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=2048)
def pauli_action(p):
    n = len(p)
    source = np.arange(2 ** n)
    destination = source.copy()
    phase = np.ones(2 ** n, dtype=complex)
    for q, a in enumerate(p):
        bit = (source >> (n - q - 1)) & 1
        if a in "XY":
            destination ^= 1 << (n - q - 1)
        if a == "Y":
            phase *= 1j * (1 - 2 * bit)
        elif a == "Z":
            phase *= 1 - 2 * bit
    return destination, phase


def apply_pauli(states, p):
    destination, phase = pauli_action(p)
    out = np.empty_like(states)
    out[:, destination] = states * phase
    return out


def apply_clifford(states, gate, n):
    if gate.op == "H":
        q = gate.wires[0]
        a = states.reshape(len(states), 2 ** q, 2, 2 ** (n - q - 1))
        out = np.empty_like(a)
        out[:, :, 0, :] = (a[:, :, 0, :] + a[:, :, 1, :]) / np.sqrt(2)
        out[:, :, 1, :] = (a[:, :, 0, :] - a[:, :, 1, :]) / np.sqrt(2)
        return out.reshape(states.shape)
    if gate.op == "CX":
        c, t = gate.wires
        source = np.arange(2 ** n)
        dest = source ^ (((source >> (n - c - 1)) & 1) << (n - t - 1))
        out = np.empty_like(states)
        out[:, dest] = states
        return out
    raise ValueError(gate.op)


def states_and_tangent(circuit, X, theta, n, parameter=None):
    X = np.asarray(X, dtype=float)
    state = np.zeros((len(X), 2 ** n), dtype=complex)
    state[:, 0] = 1
    tangent = np.zeros_like(state)
    for gate in circuit:
        if gate.op in ("H", "CX"):
            state = apply_clifford(state, gate, n)
            if parameter is not None:
                tangent = apply_clifford(tangent, gate, n)
            continue
        if gate.op != "R" or len(gate.pauli) != n:
            raise ValueError("Invalid gate")
        if gate.kind == "fixed":
            angle = np.full(len(X), gate.value)
        elif gate.kind == "data":
            angle = X[:, gate.feature]
        elif gate.kind == "param":
            angle = np.full(len(X), theta[gate.index])
        elif gate.kind == "product":
            angle = theta[gate.index] * X[:, gate.feature]
        else:
            raise ValueError(gate.kind)
        if not np.isfinite(gate.scale):
            raise ValueError('Non-finite constant angle scale')
        angle = gate.scale * angle
        co, si = np.cos(angle[:, None] / 2), np.sin(angle[:, None] / 2)
        old = state
        p_old = apply_pauli(old, gate.pauli)
        state = co * old - 1j * si * p_old
        if parameter is not None:
            tangent = co * tangent - 1j * si * apply_pauli(tangent, gate.pauli)
            if gate.parameter == parameter:
                factor = X[:, gate.feature] if gate.kind == "product" else np.ones(len(X))
                tangent += gate.scale * factor[:, None] * (-0.5 * si * old - 0.5j * co * p_old)
    return state, tangent


def kernel(circuit, X, theta, n):
    s, _ = states_and_tangent(circuit, X, theta, n)
    return np.abs(s @ s.conj().T) ** 2


def kernel_derivative(circuit, X, theta, n, parameter):
    s, t = states_and_tangent(circuit, X, theta, n, parameter)
    overlap = s @ s.conj().T
    d_overlap = t @ s.conj().T + s @ t.conj().T
    return 2 * np.real(overlap.conj() * d_overlap)


def projected_features(circuit, X, theta, n):
    s, _ = states_and_tangent(circuit, X, theta, n)
    return np.column_stack([np.real(np.sum(s.conj() * apply_pauli(s, pauli(n, "Z", q)), axis=1)) for q in range(n)])


def auc(y, score):
    """Binary AUC with average treatment of exact ties (no scipy/sklearn)."""
    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        return None
    positives, negatives = score[y], score[~y]
    delta = positives[:, None] - negatives[None, :]
    return float(np.mean((delta > 0).astype(float) + 0.5 * (delta == 0)))


def regression_metrics(K, y, train, valid, test):
    # Fixed ridge, train-only mean, validation-selected architecture. All methods
    # use identical numerical stabilization to make tie policy explicit.
    K = np.round(K, 12)
    mean = float(np.mean(y[train]))
    weights = np.linalg.solve(K[np.ix_(train, train)] + 0.1 * np.eye(len(train)), y[train] - mean)
    pv = K[np.ix_(valid, train)] @ weights + mean
    pt = K[np.ix_(test, train)] @ weights + mean
    return float(np.mean((pv - y[valid]) ** 2)), float(np.mean((pt - y[test]) ** 2))
