#!/usr/bin/env python3
"""Build/check the 54-setting evidence index; optionally replay modular minors.

Default: read-only consistency/hash check, NOT a certificate replay.
Only --build or --output writes files, always outside the public package.
"""
import argparse
import csv
from fractions import Fraction
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DEFAULT_PACKAGE = Path(os.environ.get("READOUT_KERNEL_PACKAGE",
                                     str(HERE.parents[1] / "readout_kernel_identifiability")))
COMMIT = "d53cdfe9ee254c0d687ebec28f13584218f5ff04"
CORE = "core_reproduction/core/"
PILOT = CORE + "journal_extension/dimension_pilot/"
PILOT_RUN = PILOT + "results/pilot_20261001T161006_106481Z/"
EXTENSION = CORE + "journal_extension/dimension_extension/"
GRAM = CORE + "reference_results/fourier_gram/20261002T044925Z/"
ENGINE = CORE + "journal_extension/fourier_gram/experiment/"
PRIMES = (134217689, 134217649)
PATTERN = re.compile(r"(.+)_n(\d+)_l(\d+)_(local_xyz|local_z|pauli_weight_le_2)$")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(root, relative, pointer=None):
    result = {"path": relative, "sha256": digest(root / relative)}
    if pointer:
        result["json_pointer"] = pointer
    return result


def is_prime(n):
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


def denominator_check(info, record, p):
    """Check generators of the rational denominator localization used by engine."""
    factors = [2] + [z[2] for z in record["theta_star"]["zeta"]]
    factors += [Fraction(str(w)).denominator for w in info["weights"]]
    factors += [(Fraction(str(c)) / 2).denominator
                for gate in info["program"] if gate["op"] == "R"
                for c, _, _ in gate["terms"]]
    check(all(math.gcd(f, p) == 1 for f in factors), "Noninvertible denominator")
    check(all(a*a + b*b == c*c and c > 0
              for a, b, c in record["theta_star"]["zeta"]), "Invalid rational-circle point")
    return sorted(set(factors))


def manuscript_ranks(path):
    families = {"ChebyshevRx": "chebyshev_rx", "ParamZ": "param_z_feature_map",
                "YZ-CX": "yz_cx_encoding_circuit", "Hubregtsen": "hubregtsen_encoding_circuit",
                "Kyriienko": "kyriienko_nonlinear_encoding_circuit"}
    found, family = {}, None
    for line in path.read_text().splitlines():
        cells = [v.strip() for v in line.split("&")]
        if len(cells) != 7 or not cells[1].isdigit():
            continue
        if cells[0]:
            family = families[cells[0]]
        n, layers, parameters = map(int, cells[1:4])
        for readout, value in zip(("local_xyz", "local_z", "pauli_weight_le_2"), cells[4:]):
            rank = int(re.search(r"\d+", value).group())
            found[f"{family}_n{n}_l{layers}_{readout}"] = (parameters, rank)
    return found


def build(root, manuscript):
    manifest = read(root / "MANIFEST.json")["files"]
    rows, verified = [], set()
    scopes = {PILOT: {v["id"]: v for v in read(root / (PILOT + "frozen_inputs/scope.json"))},
              EXTENSION: {v["id"]: v for v in read(root / (EXTENSION + "frozen_inputs/scope.json"))}}
    expected = set(scopes[PILOT]) | set(scopes[EXTENSION])
    check(len(scopes[PILOT]) == 24 and len(scopes[EXTENSION]) == 30 and len(expected) == 54,
          "Unexpected or overlapping census scopes")
    for folder, cohort in ((PILOT_RUN, "pilot_product_angle"), (GRAM, "extension_fixed_frequency")):
        for path in sorted((root / folder).glob("*.json")):
            rec = read(path)
            if "id" not in rec or rec["id"] not in expected:
                continue
            name = rec["id"]
            family, n, layers, readout = PATTERN.fullmatch(name).groups()
            pilot = cohort == "pilot_product_angle"
            module = PILOT if pilot else EXTENSION
            source_path = module + "frozen_inputs/" + name + ".json"
            info = read(root / source_path)
            source = artifact(root, source_path)
            check(source["sha256"] == scopes[module][name]["file_sha256"], name + ": scope source mismatch")
            source["canonical_ir_sha256"] = scopes[module][name]["source_sha256"]
            proof_path = folder + path.name
            cert = artifact(root, proof_path)
            terms = [term for gate in info["program"] if gate["op"] == "R" for term in gate["terms"]]
            product = any(t >= 0 and x >= 0 and Fraction(str(c)) for c, t, x in terms)
            check(product == pilot, name + ": product/fixed-frequency contract mismatch")
            if pilot:
                rank, upper = rec["minor"]["lower"], rec["exact"]["r_upper"]
                check(rank == upper, name + ": unclosed pilot rank")
                check(len(rec["exact"]["null_basis"]) == rec["parameter_count"] - rank,
                      name + ": null-space dimension mismatch")
                for pair in rec["pairs"]:
                    check(all(-1 < Fraction(x) < 1 for point in pair for x in point), name + ": infeasible pair")
                if rank:
                    low, high = map(Fraction, rec["minor"]["determinant_bounds"])
                    check(low <= high and (low > 0 or high < 0), name + ": zero-containing minor")
                else:
                    check(rec["exact"]["distance_is_zero"], name + ": rank-zero pilot lacks zero identity")
                active, inactive = (rec["counts"].get(k, 0) for k in ("ACTIVE_CERTIFIED", "EXACT_INACTIVE"))
                upper_source = {"method": "exact formal coefficient constant-null-space bound",
                                "artifact": artifact(root, proof_path, "/exact"),
                                "implementation": PILOT + "exact_engine.py#certify",
                                "manuscript": "dimension_appendix.tex (app:dimension)",
                                "nullity": len(rec["exact"]["null_basis"])}
                lower_source = {"method": "feasible interval Jacobian minor" if rank else "trivial nonnegative rank",
                                "artifact": artifact(root, proof_path, "/minor"),
                                "parameter_point_pointer": "/theta", "input_pairs_pointer": "/pairs",
                                "minor": rec["minor"], "generation_precision_dps": 50,
                                "public_replay_precision_dps": 65,
                                "implementation": PILOT + "run_pilot.py#audit"}
                replay_scope = "Recomputes formal upper bound and saved feasible interval witnesses at 65 dps."
                sources = [PILOT + "exact_engine.py", PILOT + "interval_engine.py", PILOT + "run_pilot.py"]
            else:
                rank, upper = rec["r_lower"], rec["r_upper"]
                check(rec["status"] == "CLOSED" and rank == upper, name + ": unclosed Gram rank")
                check(rec["fixed_frequency"] and rec["exact_supported"] and not rec["product_terms"]
                      and not rec["constant_angle_terms"] and rec["independence"]["bases_on_distinct_coordinates"],
                      name + ": finite-basis contract mismatch")
                per_prime = rec["lower_bound"]["per_prime"]
                check(tuple(v["prime"] for v in per_prime) == PRIMES, name + ": unexpected primes")
                for minor in per_prime:
                    check(minor["rank"] == rank and len(minor["minor_columns"]) == rank
                          and len(minor["minor_rows_G_entries"]) == rank, name + ": modular rank mismatch")
                    check(minor["minor_det_mod_p"] % minor["prime"] != 0,
                          name + ": zero determinant (rank zero uses empty determinant one)")
                    denominator_check(info, rec, minor["prime"])
                rational = rec["lower_bound"].get("exact_rational")
                if rational:
                    check(rational["rank"] == rank and Fraction(rational["minor_determinant"]) != 0,
                          name + ": rational minor mismatch")
                labels = [v["verdict"] for v in rec["per_symbol"]]
                active, inactive = labels.count("ACTIVE"), labels.count("INACTIVE")
                upper_source = {"method": "minimum of exact invariance-field and completed symbolic-Gram span bounds",
                                "attained_by": rec["upper_bound"]["attained_by"],
                                "artifact": artifact(root, proof_path, "/upper_bound"),
                                "implementation": ENGINE + "run_experiment.py#analyze",
                                "manuscript": "followup_appendix.tex (app:followup); fourier_gram.tex (thm:gram)",
                                "U1_bound": rec["upper_bound"]["U1"]["bound"],
                                "symbolic_original_completed": rec["upper_bound"]["U2_symbolic_gram"]["original_circuit"]["completed"]}
                lower_source = {"method": "rational Gram-Jacobian minor reduced modulo primes" if rank else "trivial nonnegative rank",
                                "artifact": artifact(root, proof_path, "/lower_bound"),
                                "parameter_point_pointer": "/theta_star", "per_prime": per_prime,
                                "rational_minor": rational,
                                "denominator_generators": denominator_check(info, rec, PRIMES[0]),
                                "implementation": ENGINE + "fourier_gram.py#LaurentEngine",
                                "homomorphism_contract": "Rational-circle phases; invertible denominator generators at each prime; reduction commutes with the rational coefficient/Jacobian computation. One nonzero reduced minor proves a rational lower bound, not an upper bound."}
                replay_scope = "Only stored Gram census aggregation in core-audit; no modular-minor or symbolic-upper recomputation."
                sources = [ENGINE + "fourier_gram.py", ENGINE + "run_experiment.py", ENGINE + "exact_engine_support/exact_engine.py"]
            check(active + inactive == rec["parameter_count"], name + ": undecided or missing activity entries")
            check(info["count"] == rec["parameter_count"] and info["n"] == int(n), name + ": input dimensions mismatch")
            for relative in [source_path, proof_path] + sources:
                check(relative in manifest and digest(root / relative) == manifest[relative], "Release manifest mismatch: " + relative)
                verified.add(relative)
            rows.append({"setting": name, "cohort": cohort, "family": family, "n": int(n), "layers": int(layers),
                         "readout": readout, "source_symbols": rec["parameter_count"], "exact_rank": rank,
                         "rank_lower": rank, "rank_upper": upper, "active": active, "inactive": inactive,
                         "positive_rank_witness": bool(rank), "frozen_input": source, "certificate": cert,
                         "upper_bound": upper_source, "lower_bound": lower_source,
                         "source_files": [artifact(root, p) for p in sources],
                         "public_audit_command": "python reproduce.py core-audit", "public_audit_scope": replay_scope,
                         "public_fresh_command": "python reproduce.py core-full --workers 2",
                         "targeted_modular_replay": None if pilot else
                         "python verify_census.py --package PACKAGE --replay-modular --setting " + name,
                         "claim_grade": "rigorous bound matching conditional on stated arithmetic/IR implementation; no QPU or independent proof-system claim"})
    check({r["setting"] for r in rows} == expected and len(rows) == 54, "Census membership mismatch")
    table = {}
    for filename in ("dimension_results.tex", "extension_results.tex"):
        table.update(manuscript_ranks(manuscript / filename))
    check(table == {r["setting"]: (r["source_symbols"], r["exact_rank"]) for r in rows}, "Manuscript ranks differ from artifacts")
    check(all(is_prime(p) for p in PRIMES), "Composite modulus")
    summary = {"settings": 54, "pilot_settings": 24, "extension_settings": 30,
               "pilot_nonzero_interval_minors": sum(bool(r["exact_rank"]) for r in rows if r["cohort"] == "pilot_product_angle"),
               "pilot_rank_zero": sum(r["exact_rank"] == 0 for r in rows if r["cohort"] == "pilot_product_angle"),
               "extension_modular_records": 60,
               "extension_nonempty_modular_minors": sum(2 for r in rows if r["cohort"] == "extension_fixed_frequency" and r["exact_rank"]),
               "extension_rank_zero": sum(r["exact_rank"] == 0 for r in rows if r["cohort"] == "extension_fixed_frequency"),
               "rational_records": sum(r["lower_bound"].get("rational_minor") is not None for r in rows),
               "nonempty_rational_minors": sum(bool(r["exact_rank"]) for r in rows if r["lower_bound"].get("rational_minor") is not None),
               "active_entries": sum(r["active"] for r in rows), "inactive_entries": sum(r["inactive"] for r in rows),
               "manifest_matched_files": len(verified)}
    return {"schema_version": 1, "public_repository": "https://github.com/parkfinity/readout_kernel_identifiability",
            "public_commit": COMMIT, "commit_note": "Commit supplied by the author; this checker verifies local release manifest bytes, not remote Git state.",
            "path_base": "public repository root", "summary": summary,
            "limits": ["Default check reads stored evidence and matches tables/hashes; it is not a fresh certificate proof.",
                       "core-audit replays pilot bounds but only aggregates the stored 30-setting Gram census.",
                       "Optional modular replay recomputes lower evidence with the released Laurent engine, not upper proofs or an independent implementation.",
                       "Rank-zero empty determinants equal one; they are not positive-rank witnesses.",
                       "The 24 product-angle settings do not use the extension's fixed finite Fourier basis theorem.",
                       "A formal identity supplies an upper bound; feasible nonzero witnesses supply lower bounds.",
                       "No QPU, hardware advantage, classification gain, or generic global coordinate minimality is certified."],
            "settings": rows}


def as_csv(mapping):
    stream = io.StringIO(newline="")
    fields = ["setting", "cohort", "family", "n", "layers", "readout", "source_symbols", "exact_rank", "active", "inactive",
              "input_artifact", "input_sha256", "certificate_artifact", "certificate_sha256", "upper_method", "upper_pointer",
              "upper_attained_by", "lower_method", "lower_pointer", "positive_rank_witness", "public_audit_command",
              "public_audit_scope", "public_fresh_command", "targeted_modular_replay"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in mapping["settings"]:
        values = {f: row[f] for f in fields if f in row}
        values.update(input_artifact=row["frozen_input"]["path"], input_sha256=row["frozen_input"]["sha256"],
                      certificate_artifact=row["certificate"]["path"], certificate_sha256=row["certificate"]["sha256"],
                      upper_method=row["upper_bound"]["method"], upper_pointer=row["upper_bound"]["artifact"]["json_pointer"],
                      upper_attained_by="; ".join(row["upper_bound"].get("attained_by", ["formal null-space identity"])),
                      lower_method=row["lower_bound"]["method"], lower_pointer=row["lower_bound"]["artifact"]["json_pointer"])
        writer.writerow(values)
    return stream.getvalue()


def replay_modular(root, rows, setting):
    sys.path.insert(0, str(root / (ENGINE + "exact_engine_support")))
    sys.path.insert(0, str(root / ENGINE))
    fg = importlib.import_module("fourier_gram")
    import numpy as np
    selected = [r for r in rows if r["cohort"] == "extension_fixed_frequency" and (not setting or r["setting"] == setting)]
    check(bool(selected), "No matching fixed-frequency setting")
    results = []
    for row in selected:
        name = row["setting"]
        record = read(root / row["certificate"]["path"])
        info = read(root / row["frozen_input"]["path"])
        st = fg.Setting(info, name)
        point = fg.pythagorean_point(st, record["theta_star"]["seed"])
        check([list(z) for z in point["zeta"]] == record["theta_star"]["zeta"], name + ": parameter point mismatch")
        minors = []
        for old in record["lower_bound"]["per_prime"]:
            p = old["prime"]
            denominator_check(info, record, p)
            # Conservative dot-product and float64 integer-bincount safety bounds.
            bound64 = max(4, 2 * st.N, len(st.obs), st.K) * (p - 1)**2
            bound53 = math.prod(st.E)**2 * (p - 1)
            check(bound64 < 2**63 and bound53 < 2**53, name + ": arithmetic safety bound failed")
            engine = fg.LaurentEngine(st, point, "mod", p).run()
            C, dC, real_ok = engine.coefficients_mod()
            J, support, iu = fg.gram_jacobian_mod(C, dC, st.w, p)
            label = lambda a: "%s(%s)" % st.basis[int(support[a])]
            index = {(label(a), label(b)): j for j, (a, b) in enumerate(zip(*iu))}
            chosen = [index[tuple(labels)] for labels in old["minor_rows_G_entries"]]
            minor = [[int(J[i, k]) for k in old["minor_columns"]] for i in chosen]
            det = fg.det_mod(minor, p)
            rank, _, _ = fg.rank_mod(J, p)
            check(real_ok and rank == old["rank"] and det == old["minor_det_mod_p"], name + ": modular replay mismatch")
            check(int(len(support)) == old["A_support"] and int(J.shape[0]) == old["jacobian_rows"], name + ": support mismatch")
            for k, witness in old["active_witnesses"].items():
                check(int(J[index[tuple(witness["row"])], int(k)]) == witness["value_mod_p"], name + ": activity mismatch")
            minors.append({"prime": p, "rank": rank, "det_mod_p": det,
                           "int64_contraction_bound": bound64, "float64_integer_binning_bound": bound53})
        results.append({"setting": name, "passed": True, "modular_certificates": minors})
        print("PASS modular replay: " + name, file=sys.stderr, flush=True)
    return {"passed": True, "settings": len(results), "modular_records": 2 * len(results),
            "numpy_version": np.__version__, "engine_sha256": digest(root / (ENGINE + "fourier_gram.py")),
            "scope": "Fresh modular coefficients/Jacobians and replay of stored minor choices/activity residues using the released implementation; no upper-proof replay or independent implementation.",
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--manuscript", type=Path, default=HERE.parent / "manuscript")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--replay-modular", action="store_true")
    parser.add_argument("--setting")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    check(__debug__, "Do not use Python -O")
    mapping = build(args.package.resolve(), args.manuscript.resolve())
    json_path, csv_path = HERE / "census_54_mapping.json", HERE / "census_54_mapping.csv"
    if args.build:
        json_path.write_text(json.dumps(mapping, indent=2) + "\n")
        with csv_path.open("w", newline="") as handle:
            handle.write(as_csv(mapping))
    else:
        check(read(json_path) == mapping, "Stored JSON mapping differs from source artifacts")
        with csv_path.open(newline="") as handle:
            check(handle.read() == as_csv(mapping), "Stored CSV mapping differs from source artifacts")
    verdict = {"passed": True, "mode": "mapping_consistency_not_certificate_replay", **mapping["summary"]}
    if args.replay_modular:
        verdict["modular_replay"] = replay_modular(args.package.resolve(), mapping["settings"], args.setting)
    if args.output:
        check(HERE in args.output.resolve().parents, "Output must remain inside census_mapping/")
        args.output.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
