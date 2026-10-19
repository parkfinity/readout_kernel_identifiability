"""Frozen, untuned minimal-product sensitivity comparison (NumPy only)."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parents[1] / "quantum_public_release_20261018/readout_kernel_identifiability"
BUDGETS = [256, 1024, 4096, 16384]
ESTIMATORS = ["direct_parity", "product_marginals"]
EPSILON = 0.1
REPLICATES = 256
RANK_RTOLS = [1e-8, 1e-9, 1e-10]
BITS = 1 - 2 * ((np.arange(16)[:, None] >> np.arange(4)) & 1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n")


def designs():
    local = [[0], [1], [2], [3]]
    w = np.arange(1.0, 5.0)
    one = np.eye(5)
    one[:2, :2] = [[2, 1], [1, 2]]
    pairs = [list(p) for p in itertools.combinations(range(4), 2)]
    matrices = [(32*np.eye(4) + np.outer(w, w))/158, one/7]
    matrices += [np.diag(np.linspace(1, 4, n))/np.linspace(1, 4, n).sum() for n in [6, 10]]
    sets = [local, local + [[0, 2]], local + [[0, 2], [1, 3]], local + pairs]
    return [dict(name=name, products=k, sets=s, metric=m.tolist(),
                 metric_condition_number=float(np.linalg.cond(m)))
            for name, k, s, m in zip(["zero_dense", "one_dense", "two_diagonal", "six_diagonal"],
                                    [0, 1, 2, 6], sets, matrices)]


def freeze(source):
    """Extract only the pre-existing test points; never evaluate design outcomes."""
    manifest_path = HERE / "FROZEN_MANIFEST.json"
    if manifest_path.exists() or (HERE / "frozen_inputs.json").exists():
        raise SystemExit("Refuse to overwrite an existing freeze; use a new version.")
    source = Path(source)
    public = source / "experiments/finite_shot"
    historic = source / "provenance/historical_records/experiments/finite_shot"
    plan_path = historic / "plan.json"
    program_path = public / "frozen_inputs/hubregtsen_encoding_circuit_n4_l1_local_z.json"
    original_manifest = json.loads((public / "FROZEN_MANIFEST.json").read_text())
    assert sha(plan_path) == original_manifest["plan.json"]
    assert sha(program_path) == original_manifest["frozen_inputs/" + program_path.name]
    plan = json.loads(plan_path.read_text())
    points = []
    for panel in plan["panels"]:
        for point, rec in enumerate(panel["groups"]["hub_z"]["test"]):
            points.append(dict(panel=panel["panel"], point=point,
                               original_generation_seed=2026100521+100*panel["panel"],
                               theta=rec["theta"], x=rec["x"], phi=rec["phi"],
                               Q=np.eye(4).tolist()))
    assert len(points) == 15
    frozen = dict(version=1, frozen_utc=datetime.now(timezone.utc).isoformat(),
                  status="Informed follow-up; existing held-out points reused, not new data or preregistration",
                  source_program=json.loads(program_path.read_text()),
                  source_plan_relative=str(plan_path.relative_to(source)),
                  source_program_relative=str(program_path.relative_to(source)),
                  original_plan_sha256=sha(plan_path), original_program_sha256=sha(program_path),
                  original_manifest_sha256=sha(public / "FROZEN_MANIFEST.json"),
                  source_public_code_sha256={name: sha(public/name) for name in ["core.py", "legacy_design.py", "prepare.py"]},
                  points=points, designs=designs(), target_coordinates=["theta0", "theta1", "theta2", "theta3"],
                  target_basis=np.eye(4).tolist(), epsilon=EPSILON, budgets=BUDGETS,
                  estimators=ESTIMATORS, replicates=REPLICATES, rng_root_seed=2026101901,
                  rank_relative_tolerances=RANK_RTOLS, rank_absolute_floor=1e-11)
    write_json(HERE/"frozen_inputs.json", frozen)
    names = ["PROTOCOL.txt", "run.py", "test_experiment.py", "frozen_inputs.json"]
    write_json(manifest_path, {name: sha(HERE/name) for name in names})
    print(json.dumps(dict(frozen=True, points=len(points), manifest_sha256=sha(manifest_path)), indent=2))


def load_frozen():
    for name, digest in json.loads((HERE/"FROZEN_MANIFEST.json").read_text()).items():
        if sha(HERE/name) != digest:
            raise ValueError("Frozen file changed: " + name)
    return json.loads((HERE/"frozen_inputs.json").read_text())


def local_model(theta, x):
    """Output layout: input x qubit, input x qubit x original angle."""
    angles = np.asarray(theta)[:4]
    uvuv = np.cos(np.asarray(x))[:, [0, 1, 0, 1]]
    mu = -np.sin(angles)[None, :]*uvuv
    dmu = np.zeros((len(mu), 4, 4))
    dmu[:, np.arange(4), np.arange(4)] = -np.cos(angles)[None, :]*uvuv
    return mu, dmu


def features(mu, dmu, sets):
    f = np.stack([np.prod(mu[:, s], axis=1) for s in sets], axis=1)
    df = np.zeros((len(mu), len(sets), 4))
    for i, s in enumerate(sets):
        for q in s:
            others = [r for r in s if r != q]
            df[:, i, :] += np.prod(mu[:, others], axis=1)[:, None]*dmu[:, q, :]
    return f, df


def feature_covariance(mu, sets, half, estimator):
    means = np.stack([np.prod(mu[:, s], axis=1) for s in sets], axis=1)
    second = mu**2 + (1-mu**2)/half
    cov = np.empty((len(mu), len(sets), len(sets)))
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            odd = sorted(set(a)^set(b))
            if estimator == "direct_parity":
                cov[:, i, j] = (np.prod(mu[:, odd], axis=1)-means[:, i]*means[:, j])/half
            elif estimator == "product_marginals":
                common = sorted(set(a)&set(b))
                moment = np.prod(second[:, common], axis=1)*np.prod(mu[:, odd], axis=1)
                cov[:, i, j] = moment-means[:, i]*means[:, j]
            else:
                raise ValueError(estimator)
    return cov


def distance_variance(delta, cov, metric):
    pairs = len(delta)
    S = cov[:pairs] + cov[pairs:]
    md = delta @ metric
    return 2*np.einsum("pi,pij,pj->p", md, S, md) + np.einsum("ij,pjk,kl,pli->p", metric, S, metric, S)


def rank_diagnostics(jac):
    sv = np.linalg.svd(jac, compute_uv=False)
    ranks = [int(np.sum(sv > max(1e-11, t*sv[0]))) for t in RANK_RTOLS]
    return sv, ranks


def sensitivity(jac, variance, rank):
    if not np.all(variance > 0):
        raise ValueError("All distance variances must be strictly positive")
    whitened = EPSILON * jac / np.sqrt(2*variance)[:, None]
    sv = np.linalg.svd(whitened, compute_uv=False)
    return dict(whitened_jacobian=whitened, whitened_singular_values=sv,
                retained_sensitivity=float(sv[rank-1]) if rank else 0.0,
                whole_space_weakest=float(sv[3]) if rank == 4 else 0.0)


def probabilities(mu):
    return np.prod((1 + mu[:, None, :]*BITS[None, :, :])/2, axis=2)


def pauli_matrix(word):
    pauli = dict(I=np.eye(2), X=np.array([[0, 1], [1, 0]]),
                 Y=np.array([[0, -1j], [1j, 0]]), Z=np.diag([1, -1]))
    out = np.ones((1, 1), complex)
    for letter in word[::-1]:
        out = np.kron(out, pauli[letter])
    return out


def dense_source(source, theta, x):
    """Independent statevector propagation of actual frozen source gate IR."""
    x = np.asarray(x)
    raw = {"a0": x[:, 0], "a1": x[:, 1], "acos(a0)": np.arccos(x[:, 0]),
           "acos(a1)": np.arccos(x[:, 1])}
    phi = np.stack([raw[s] for s in source["data_expressions"]], axis=1)
    n = source["n"]
    psi = np.zeros((len(x), 2**n), complex)
    psi[:, 0] = 1
    for gate in source["program"]:
        if gate["op"] == "H":
            q = gate["wires"][0]
            px = "I"*q + "X" + "I"*(n-q-1)
            pz = "I"*q + "Z" + "I"*(n-q-1)
            psi = psi @ ((pauli_matrix(px)+pauli_matrix(pz))/np.sqrt(2)).T
        elif gate["op"] == "R":
            angle = np.zeros(len(x))
            for coeff, t, d in gate["terms"]:
                if t >= 0 and d >= 0:
                    angle += coeff*theta[t]*phi[:, d]
                elif t >= 0:
                    angle += coeff*theta[t]
                elif d >= 0:
                    angle += coeff*phi[:, d]
                else:
                    angle += coeff
            psi = np.cos(angle/2)[:, None]*psi - 1j*np.sin(angle/2)[:, None]*(psi @ pauli_matrix(gate["pauli"]).T)
        else:
            raise ValueError("Unsupported source gate: " + gate["op"])
    return psi


def source_check(frozen, point):
    theta, x = np.array(point["theta"]), np.array(point["x"])
    source = frozen["source_program"]
    assert source["n"] == 4 and source["count"] == 8
    assert source["data_expressions"] == ["a0", "a1", "a0", "a1"]
    # The public Fourier engine canonicalizes repeated data angles to two slots.
    assert np.array_equal(x, np.array(point["phi"]))
    assert np.array_equal(point["Q"], np.eye(4))
    mu, dmu = local_model(theta, x)
    psi = dense_source(source, theta, x)
    pr = abs(psi)**2
    all_sets = frozen["designs"][-1]["sets"]
    signs = np.stack([np.prod(BITS[:, s], axis=1) for s in all_sets], axis=1)
    f, df = features(mu, dmu, all_sets)
    errors = dict(probability_factorization=float(np.max(abs(pr-probabilities(mu)))),
                  ideal_features=float(np.max(abs(pr@signs-f))),
                  probability_normalization=float(np.max(abs(pr.sum(axis=1)-1))))
    fd = np.empty_like(df)
    tail_error = 0.0
    for q in range(8):
        step = np.eye(8)[q]*1e-5
        diff = ((abs(dense_source(source, theta+step, x))**2)@signs -
                (abs(dense_source(source, theta-step, x))**2)@signs)/(2e-5)
        if q < 4:
            fd[:, :, q] = diff
        else:
            tail_error = max(tail_error, float(np.max(abs(diff))))
    errors["angle_derivative_finite_difference"] = float(np.max(abs(fd-df)))
    errors["tail_derivative_finite_difference"] = tail_error
    if max(errors.values()) > 1e-8:
        raise AssertionError(errors)
    return errors


def shot_counts(pr, half, reps, rng):
    """Same full bitstring counts feed every design and both estimators."""
    return np.stack([np.stack([rng.multinomial(half, row, size=reps)
                               for row in pr], axis=1) for _ in range(2)])


def sample_features(counts, sets, half, estimator):
    if estimator == "direct_parity":
        signs = np.stack([np.prod(BITS[:, s], axis=1) for s in sets], axis=1)
        return counts @ signs / half
    local = counts @ BITS / half
    return np.stack([np.prod(local[:, :, :, s], axis=3) for s in sets], axis=3)


def sample_distances(f, metric):
    pairs = f.shape[2]//2
    difference = f[:, :, :pairs, :] - f[:, :, pairs:, :]
    return np.einsum("rpi,ij,rpj->rp", difference[0], metric, difference[1])


def summaries(records):
    panels = []
    for panel in range(3):
        for budget in BUDGETS:
            for design in [d["name"] for d in designs()]:
                for estimator in ESTIMATORS:
                    rows = [r for r in records if (r["panel"], r["budget"], r["design"], r["estimator"]) == (panel, budget, design, estimator)]
                    weakest = [r["whole_space_weakest"] for r in rows]
                    panels.append(dict(panel=panel, budget=budget, design=design, estimator=estimator,
                                       points=len(rows), ranks=sorted(set(r["rank"] for r in rows)),
                                       whole_space_weakest_median=float(np.median(weakest)),
                                       whole_space_weakest_range=[min(weakest), max(weakest)],
                                       retained_sensitivity_median=float(np.median([r["retained_sensitivity"] for r in rows]))))
    comparisons = []
    by = {(r["panel"], r["point"], r["budget"], r["design"], r["estimator"]): r for r in records}
    for panel in range(3):
        for budget in BUDGETS:
            for estimator in ESTIMATORS:
                for baseline in ["two_diagonal", "six_diagonal"]:
                    ratios = [by[panel, p, budget, "one_dense", estimator]["whole_space_weakest"] /
                              by[panel, p, budget, baseline, estimator]["whole_space_weakest"] for p in range(5)]
                    comparisons.append(dict(panel=panel, budget=budget, estimator=estimator,
                                            numerator="one_dense", denominator=baseline, paired_ratios=ratios,
                                            median=float(np.median(ratios)), range=[min(ratios), max(ratios)],
                                            above_one=sum(v > 1 for v in ratios), below_one=sum(v < 1 for v in ratios)))
    return panels, comparisons


def latex(summary):
    rows = summary["panel_summaries"]
    def val(name, estimator, panel):
        return next(r["whole_space_weakest_median"] for r in rows if
                    (r["design"], r["estimator"], r["panel"], r["budget"]) == (name, estimator, panel, 4096))
    table = []
    for design in designs():
        direct = ",".join("%.4g" % val(design["name"], ESTIMATORS[0], p) for p in range(3))
        product = ",".join("%.4g" % val(design["name"], ESTIMATORS[1], p) for p in range(3))
        table.append("%d & %s & $%s$ & $%s$ \\\\" % (design["products"], "3/4" if design["products"] == 0 else "4/4", direct, product))
    text = r"""% Generated by experiment/run.py; every point and budget retained in results.
\subsection{Matched finite-shot check of minimal product recovery}
\label{sec:minimalproductshots}
We compare four untuned, unit-trace designs on the same four-qubit Hubregtsen
source: no product with $M=(32I+ww^T)/158$, $w=(1,2,3,4)^T$; one product
$f_0f_2$ with $M=(\left(\begin{smallmatrix}2&1\\1&2\end{smallmatrix}\right)
\oplus I_3)/7$ in order $(f_0,f_1,f_2,f_3,f_0f_2)$; two products
$(f_0f_2,f_1f_3)$ and all six products, with diagonal weights evenly spaced
from 1 to 4 and normalized to unit trace. The latter two use the earlier
fixed-diagonal convention. Thus the comparison is between complete fixed
designs, not a feature-count-only effect.

This informed follow-up reuses all three previously frozen panels and their
five held-out points, with 16 disjoint input pairs per point. It is not new
held-out data or independent preregistration. No metric selection is performed.
The target is the original four-angle Euclidean space, $Q=I_4$, with
$\epsilon=0.1$ in Eq.~\eqref{eq:shotwhiten}. All designs use the same one-basis
ZZZZ outcomes, split into two independent halves, at each of the four budgets
$256,1024,4096,16384$ per input and parameter state. Both direct-parity and
product-of-marginal estimators use the full exact covariance; the latter is
unbiased only under the verified factorized Z-distribution. No extra shots
are allocated to extra products.

\begin{table}[htbp]
\centering
\caption{Whole-space weakest noise-normalized sensitivity at 4,096 shots.
Each triple contains the three panel medians over five matched points. The
no-product control has a missing fourth direction, so its whole-space value
is zero rather than its positive smallest retained value. These are local
exact-model diagnostics, not detection probabilities or QPU results.}
\label{tab:minimalproductshots}
\small
\begin{tabular}{@{}rccl@{}}
\toprule
Products & Rank/target & Direct parity & Product marginals\\
\midrule
""" + "\n".join(table) + r"""
\bottomrule
\end{tabular}
\end{table}

All 15 points give numerical ranks three without products and four for the
three lifted designs at the three recorded tolerances. This checks the
separate analytic result but is not itself an exact rank certificate.
One-product minimality does not imply superior finite-shot conditioning:
the full record retains both improvements and deteriorations relative to
the two- and six-product designs. For a fixed design, product estimation
never increases the exact distance variance and never decreases an ordered
singular value of the noise-whitened Jacobian, within numerical tolerance.
The frozen protocol, complete 480 point/design/estimator/budget records,
exact covariances, Jacobians, and 256-replicate paired sanity checks are
included in the experiment bundle. All budgets are reported there; no
threshold-based required-shot or hardware-saving claim is made.
"""
    return text


def run():
    start = time.perf_counter()
    frozen = load_frozen()
    out = HERE/"results"
    out.mkdir(exist_ok=False)
    records, checks, covariances, samples = [], [], {}, {}
    variance_monotonicity, singular_monotonicity = [], []
    for point in frozen["points"]:
        panel, p = point["panel"], point["point"]
        errors = source_check(frozen, point)
        checks.append(dict(panel=panel, point=p, **errors))
        mu, dmu = local_model(point["theta"], point["x"])
        pairs = len(mu)//2
        for budget in frozen["budgets"]:
            half = budget//2
            rng = np.random.default_rng(np.random.SeedSequence([frozen["rng_root_seed"], panel, p, budget]))
            counts = shot_counts(probabilities(mu), half, frozen["replicates"], rng)
            for design in frozen["designs"]:
                sets, metric = design["sets"], np.array(design["metric"])
                f, df = features(mu, dmu, sets)
                delta = f[:pairs]-f[pairs:]
                ddelta = df[:pairs]-df[pairs:]
                jac = 2*np.einsum("pi,ij,pjk->pk", delta, metric, ddelta)
                sv, ranks = rank_diagnostics(jac)
                ideal = np.einsum("pi,ij,pj->p", delta, metric, delta)
                if len(set(ranks)) != 1:
                    raise AssertionError("Rank unstable across tolerances")
                expected = 3 if design["products"] == 0 else 4
                if ranks[1] != expected:
                    raise AssertionError((panel, p, design["name"], ranks))
                paired = []
                for estimator in frozen["estimators"]:
                    cov = feature_covariance(mu, sets, half, estimator)
                    variance = distance_variance(delta, cov, metric)
                    stats = sensitivity(jac, variance, ranks[1])
                    draws = sample_distances(sample_features(counts, sets, half, estimator), metric)
                    mean, empvar = draws.mean(axis=0), draws.var(axis=0, ddof=1)
                    name = "p%d_t%d_b%d_%s_%s" % (panel, p, budget, design["name"], estimator)
                    covariances[name] = cov
                    samples[name] = draws
                    row = dict(panel=panel, point=p, budget=budget, design=design["name"], estimator=estimator,
                               products=design["products"], rank=ranks[1], ranks_by_tolerance=ranks,
                               target_dimension=4, original_angle_jacobian=jac, singular_values=sv,
                               exact_pair_distances=ideal, exact_pair_variances=variance,
                               half_shots=half, actual_shots_per_input_state=2*half,
                               actual_shots_all_pairs_one_state=2*pairs*budget,
                               actual_shots_all_pairs_two_states=4*pairs*budget,
                               replicates=frozen["replicates"], empirical_pair_means=mean,
                               empirical_pair_variances=empvar, empirical_exact_variance_ratios=empvar/variance,
                               mean_bias_mc_se=(mean-ideal)/np.sqrt(empvar/frozen["replicates"]),
                               covariance_archive_key=name, samples_archive_key=name, **stats)
                    records.append(row)
                    paired.append(row)
                direct, product = paired
                variance_monotonicity.extend((np.array(product["exact_pair_variances"])/direct["exact_pair_variances"]).tolist())
                singular_monotonicity.extend((np.array(product["whitened_singular_values"])[:expected] /
                                             direct["whitened_singular_values"][:expected]).tolist())
        print("panel %d point %d: source checked; all designs, budgets and estimators complete" % (panel, p), flush=True)
    if max(variance_monotonicity) > 1+1e-9 or min(singular_monotonicity) < 1-1e-9:
        raise AssertionError("Same-design product-estimator monotonicity violated")
    panels, comparisons = summaries(records)
    calibration = np.concatenate([r["empirical_exact_variance_ratios"] for r in records])
    summary = dict(complete=True, scope="Local ideal-model fixed-budget sensitivity; no detection/QPU/shot-saving claim",
                   record_count=len(records), panels=3, points=15, pairs_per_point=16,
                   replicates=frozen["replicates"], budgets=frozen["budgets"],
                   manifest_sha256=sha(HERE/"FROZEN_MANIFEST.json"),
                   frozen_inputs_sha256=sha(HERE/"frozen_inputs.json"),
                   source_check_max_errors={k: max(c[k] for c in checks) for k in checks[0] if k not in ["panel", "point"]},
                   product_direct_pair_variance_ratio_range=[min(variance_monotonicity), max(variance_monotonicity)],
                   product_direct_retained_ordered_singular_ratio_range=[min(singular_monotonicity), max(singular_monotonicity)],
                   empirical_exact_variance_ratio_quantiles=dict(zip(["q05", "median", "q95"], np.quantile(calibration, [.05, .5, .95]))),
                   maximum_absolute_bias_mc_se=float(max(np.max(abs(r["mean_bias_mc_se"])) for r in records)),
                   panel_summaries=panels, paired_comparisons=comparisons,
                   environment=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(), executable=sys.executable),
                   elapsed_seconds=time.perf_counter()-start)
    write_json(out/"records.json", records)
    write_json(out/"source_checks.json", checks)
    write_json(out/"summary.json", summary)
    np.savez_compressed(out/"exact_half_feature_covariances.npz", **covariances)
    np.savez_compressed(out/"paired_distance_samples.npz", **samples)
    (out/"minimal_product_results.tex").write_text(latex(summary))
    write_json(out/"RESULT_MANIFEST.json", {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({k: v for k, v in summary.items() if k not in ["panel_summaries", "paired_comparisons"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run"])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    freeze(args.source) if args.action == "freeze" else run()
