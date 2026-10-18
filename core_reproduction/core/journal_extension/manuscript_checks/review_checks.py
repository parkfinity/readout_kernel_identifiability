"""Exact counterexample and reaggregation of archived controls, not new training.

Run with Python 3.9+ and SymPy. Inputs are immutable copies of archived CSVs.
Only analysis/review_checks.json is generated. No hardware is used.
"""
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "analysis/review_inputs"


def read_csv(name):
    with (INPUTS / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def exact_counterexample():
    t, x, y = sp.symbols("theta x xprime", real=True)
    psi = sp.Matrix([sp.cos((t + x) / 2), sp.sin((t + x) / 2)])
    other = psi.subs(x, y)
    tangent = psi.diff(t)
    qfi = sp.trigsimp(4 * (tangent.dot(tangent) - psi.dot(tangent) ** 2))
    fidelity = sp.trigsimp(psi.dot(other) ** 2)
    features = sp.Matrix([sp.sin(t + x), 0, sp.cos(t + x)])
    delta = features - features.subs(x, y)
    xyz_distance = sp.trigsimp(sp.expand_trig(delta.dot(delta)))
    z_distance = (sp.cos(t + x) - sp.cos(t + y)) ** 2
    z_witness = sp.simplify(z_distance.diff(t).subs({t: 0, x: 0, y: sp.pi/2}))
    assert qfi == 1
    assert sp.trigsimp(fidelity - sp.cos((x - y) / 2) ** 2) == 0
    assert sp.trigsimp(fidelity.diff(t)) == 0
    assert sp.trigsimp(xyz_distance - 2 + 2 * sp.cos(x - y)) == 0
    assert sp.trigsimp(xyz_distance.diff(t)) == 0
    assert z_witness == 2
    return {
        "state": "Ry(theta) Ry(x) |0>",
        "state_SLD_QFI": str(qfi),
        "real_QGT": str(qfi / 4),
        "fidelity_kernel": str(fidelity),
        "XYZ_distance_squared": str(xyz_distance),
        "fidelity_derivative": "0 identically",
        "XYZ_distance_derivative": "0 identically",
        "Z_distance_derivative_at_theta0_x0_xprime_pi_over_2": str(z_witness),
        "passed": True,
    }


def summarize_controls(name, accuracy_key):
    rows = read_csv(name)
    groups = defaultdict(list)
    identities = {(r["dataset"], r["model"], int(r["seed"])) for r in rows}
    assert len(rows) == len(identities) == 40
    for row in rows:
        groups[(row["dataset"], row["model"])].append(row)
    summary = []
    for (dataset, model), group in sorted(groups.items()):
        assert sorted(int(r["seed"]) for r in group) == list(range(10))
        values = [float(r[accuracy_key]) for r in group]
        entry = {
            "dataset": dataset, "model": model, "n_trials": len(group),
            "kta_before_mean": statistics.mean(float(r["kta_before"]) for r in group),
            "kta_after_mean": statistics.mean(float(r["kta_after"]) for r in group),
            "accuracy_after_mean": statistics.mean(values),
            "accuracy_after_sample_sd": statistics.stdev(values),
        }
        if "gradient_norm" in group[0]:
            entry["initial_gradient_norm_mean"] = statistics.mean(float(r["gradient_norm"]) for r in group)
            assert all((int(r["n_parameters"]), int(r["n_data_blocks"]), int(r["n_parameter_blocks"])) == (20, 3, 2) for r in group)
        summary.append(entry)
    return {"row_count": len(rows), "training_rerun": False, "groups": summary}


def summarize_phase_a():
    rows = read_csv("phase_a_candidate_results.csv")
    candidates = {r["circuit_id"] for r in rows}
    cache_files = {r["cache_file"] for r in rows}
    groups = {(r["cache_file"], r["embedding"]) for r in rows}
    assert len(rows) == 1692 and len(candidates) == 36 and len(cache_files) == 47
    assert len({(r["cache_file"], r["circuit_id"]) for r in rows}) == len(rows)
    assert all((int(r["n_support"]), int(r["n_query"])) == (4, 20) for r in rows)
    assert len(groups) == 47 * 4
    mismatches = sum(float(r["stabilized_auroc"]) != float(r["canonical_stabilized_auroc"]) for r in rows)
    assert mismatches == 0
    return {
        "rows": len(rows), "categories": len(cache_files),
        "raw_candidates": len(candidates), "conservative_embedding_keys": 4,
        "minimal_inequivalent_class_count_proven": False,
        "candidate_reduction_fraction": 1 - 4/36,
        "unbatched_pair_tasks_before": len(rows) * 4 * 20,
        "unbatched_pair_tasks_after": len(groups) * 4 * 20,
        "max_kernel_difference": max(float(r["max_kernel_difference"]) for r in rows),
        "stabilized_AUROC_mismatches": mismatches,
        "hardware_experiment": False, "circuit_replay_rerun": False,
        "scope": "Archived terminal-heavy fidelity candidates; not an adaptive PQK search or new-baseline increment.",
    }


def main():
    result = {
        "scope": "New symbolic sanity check and archived CSV reaggregation; no new model training or QPU experiment.",
        "sympy_version": sp.__version__,
        "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUTS.glob("*.csv"))},
        "QFI_counterexample": exact_counterexample(),
        "matched_classical": summarize_controls("wivf_exact_matched_results.csv", "acc_after"),
        "matched_TypeII_III": summarize_controls("y92x_matched_typeii_typeiii_results.csv", "accuracy_after"),
        "phase_a": summarize_phase_a(),
    }
    path = ROOT / "analysis/review_checks.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
