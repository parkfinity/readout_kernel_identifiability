"""Reproduce the previously omitted (n,k)=(5,3) numerical rank scan."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'archived_code'))
import fast_sim as legacy
import kernel_rank_numerics as dense


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def masks(words):
    indices = np.arange(2 ** len(words[0]))
    destinations, coefficients = [], []
    for word in words:
        x, z, _ = legacy.pauli_masks(word)
        # Matrix-row convention: Y has row coefficients (-i,+i).
        phase = (-1j) ** word.count('Y')
        signs = np.array([(-1.) ** bin(int(i) & z).count('1') for i in indices])
        destinations.append(indices ^ x)
        coefficients.append(phase * signs)
    return np.array(destinations), np.array(coefficients)


def masked_features(states, lookup):
    destinations, coefficients = lookup
    return np.einsum('bi,bmi,mi->bm', states.conj(), states[:, destinations],
                     coefficients, optimize=True).real


def features(theta, inputs, n, lookup):
    states = np.array([legacy.build_state(n, x, theta) for x in inputs])
    return masked_features(states, lookup)


def distance_and_jacobian(theta, inputs, feature_function):
    p = len(inputs) // 2
    base = feature_function(theta, inputs)
    delta = base[:p] - base[p:]
    jacobian = np.empty((p, len(theta)))
    for j in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += np.pi / 2
        minus[j] -= np.pi / 2
        derivative = (feature_function(plus, inputs) - feature_function(minus, inputs)) / 2
        # Shift the EXPECTATIONS, then differentiate the quadratic distance.
        jacobian[:, j] = 2 * np.sum(delta * (derivative[:p] - derivative[p:]), axis=1)
    return base, np.sum(delta ** 2, axis=1), jacobian


def numeric_rank(singular_values, relative, absolute):
    return int(np.count_nonzero(singular_values > max(relative * singular_values[0], absolute)))


def source_model(n):
    host = ROOT / 'core_reproduction/core/journal_extension/kyriienko_scaling/core_copy'
    sys.path.insert(0, str(host))
    import source_host
    from qiskit.quantum_info import Statevector, Pauli
    entry = next(e for e in source_host.inventory()
                 if e['module'] == 'kyriienko_nonlinear_encoding_circuit')
    obj, _, _, parameters, _ = source_host.construct(entry, n, 1)
    assert len(parameters) == 3 * n
    assert obj.encoding_style == 'chebyshev_tower' and obj.rotation_gate == 'ry'
    assert obj.variational_arrangement == 'HEA'
    assert obj.num_encoding_layers == obj.num_variational_layers == 1
    assert obj.alpha == 2.0 and obj.num_chebyshev == n
    words = legacy.readout(n, 'weight_le_3')
    # Qiskit labels qubit zero at the RIGHT, while our strings use the LEFT.
    paulis = [Pauli(word[::-1]) for word in words]

    def source_features(theta, inputs):
        result = []
        for x in inputs:
            state = Statevector.from_instruction(obj.get_circuit(x, theta))
            result.append([state.expectation_value(op).real for op in paulis])
        return np.array(result)
    return source_features, host / entry['path'], host / 'source_host.py'


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = json.loads((HERE / 'protocol.json').read_text())
    n, k = protocol['n'], protocol['k']
    words = legacy.readout(n, 'weight_le_' + str(k))
    assert len(words) == protocol['readout_count'] == 375
    lookup = masks(words)
    ops = np.array([dense.pauli_string_op(word) for word in words])
    fast = lambda theta, inputs: features(theta, inputs, n, lookup)

    def dense_features(theta, inputs):
        states = np.array([dense.build_state(n, x, theta) for x in inputs])
        return np.einsum('bi,mij,bj->bm', states.conj(), ops, states, optimize=True).real

    source, source_path, host_path = source_model(n)
    import qiskit
    results = []
    arrays = {}
    max_errors = {'dense_features': 0., 'dense_jacobian': 0., 'source_features': 0.,
                  'source_distance': 0., 'source_jacobian': 0., 'finite_difference': 0.}
    for seed in protocol['seeds']:
        tick = time.monotonic()
        rng = np.random.default_rng(seed)
        theta = rng.uniform(-2*np.pi, 2*np.pi, size=3*n)
        p = protocol['input_pairs_per_seed']
        x = rng.uniform(*protocol['input_range'], size=(p, 2))
        xp = rng.uniform(*protocol['input_range'], size=(p, 2))
        inputs = np.concatenate([x, xp])
        base, distances, jacobian = distance_and_jacobian(theta, inputs, fast)
        dense_base, _, dense_jac = distance_and_jacobian(theta, inputs, dense_features)
        s = np.linalg.svd(jacobian, compute_uv=False)
        ds = np.linalg.svd(dense_jac, compute_uv=False)
        errors = dict(dense_features=float(np.max(abs(base-dense_base))),
                      dense_jacobian=float(np.max(abs(jacobian-dense_jac))))
        count = protocol['source_crosscheck_pairs_per_seed']
        subset = np.concatenate([x[:count], xp[:count]])
        sb, sd, sj = distance_and_jacobian(theta, subset, source)
        indices = np.r_[np.arange(count), p+np.arange(count)]
        errors.update(source_features=float(np.max(abs(sb-base[indices]))),
                      source_distance=float(np.max(abs(sd-distances[:count]))),
                      source_jacobian=float(np.max(abs(sj-jacobian[:count]))))
        h = protocol['finite_difference_step']
        finite = np.empty((count, 3*n))
        for j in range(3*n):
            tp, tm = theta.copy(), theta.copy()
            tp[j] += h
            tm[j] -= h
            fp, fm = fast(tp, subset), fast(tm, subset)
            finite[:, j] = (np.sum((fp[:count]-fp[count:])**2, axis=1)
                             - np.sum((fm[:count]-fm[count:])**2, axis=1))/(2*h)
        errors['finite_difference'] = float(np.max(abs(finite-jacobian[:count])))
        for key, error in errors.items():
            max_errors[key] = max(max_errors[key], error)
            limit = protocol['finite_difference_tolerance'] if key == 'finite_difference' else protocol['crosscheck_tolerance']
            if error > limit:
                raise AssertionError((seed, key, error, limit))
        ranks = {str(t): numeric_rank(s, t, protocol['absolute_tolerance'])
                 for t in protocol['relative_tolerances']}
        dense_ranks = {str(t): numeric_rank(ds, t, protocol['absolute_tolerance'])
                       for t in protocol['relative_tolerances']}
        assert ranks == dense_ranks
        record = dict(seed=seed, rank=numeric_rank(s, protocol['primary_relative_tolerance'],
                      protocol['absolute_tolerance']), ranks_by_relative_tolerance=ranks,
                      dense_ranks_by_relative_tolerance=dense_ranks,
                      singular_values=s.tolist(), crosscheck_errors=errors,
                      elapsed_seconds=time.monotonic()-tick)
        results.append(record)
        for key, value in dict(theta=theta, x=x, xprime=xp, features=base,
                               distances=distances, jacobian=jacobian,
                               dense_jacobian=dense_jac, singular_values=s,
                               source_features=sb, source_jacobian=sj).items():
            arrays['seed_' + str(seed) + '_' + key] = value
        print(json.dumps(record), flush=True)
    # Historical implementation uses a signed Pauli-coordinate convention.
    # Odd-Y sign flips cancel exactly in unit-weight squared distances.
    old_rank, old_s, old_m = legacy.kernel_rank(n, 'weight_le_3',
                                              protocol['input_pairs_per_seed'], 3)
    error = float(np.max(abs(old_s-np.array(results[0]['singular_values']))))
    assert old_m == len(words) and old_rank == results[0]['rank']
    assert error < protocol['crosscheck_tolerance']
    np.savez_compressed(output / 'arrays.npz', **arrays)
    save(output / 'protocol.json', protocol)
    source_files = [HERE/'run.py', HERE/'protocol.json', HERE/'test_weight3.py',
                    HERE/'archived_code/fast_sim.py', HERE/'archived_code/kernel_rank_numerics.py',
                    source_path, host_path]
    report = dict(passed=True, scope=protocol['scope'], protocol=protocol, settings=results,
                  max_errors=max_errors, legacy_seed3_rank=old_rank,
                  legacy_singular_values_max_error=error,
                  sources={str(p.relative_to(ROOT)): sha(p) for p in source_files},
                  arrays_sha256=sha(output/'arrays.npz'),
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   qiskit=qiskit.__version__, platform=platform.platform()),
                  elapsed_seconds=time.monotonic()-started)
    save(output/'results.json', report)
    print(json.dumps(dict(passed=True, ranks=[r['rank'] for r in results],
                         max_errors=max_errors, output=str(output))), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory; existing results are never overwritten.')
    args = parser.parse_args()
    run(args.output)
