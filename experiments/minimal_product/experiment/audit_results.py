"""Post-run, read-only scientific consistency audit and unit-test evidence."""
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import run as experiment


def main():
    frozen = experiment.load_frozen()
    out = experiment.HERE / "results"
    for name, digest in json.loads((out/"RESULT_MANIFEST.json").read_text()).items():
        assert experiment.sha(out/name) == digest, name
    records = json.loads((out/"records.json").read_text())
    summary = json.loads((out/"summary.json").read_text())
    assert len(records) == 480
    checks = dict(max_covariance_asymmetry=0.0, minimum_covariance_eigenvalue=1.0,
                  max_variance_recomputed_error=0.0, max_sample_mean_recomputed_error=0.0,
                  max_sample_variance_recomputed_error=0.0, max_whitening_recomputed_error=0.0)
    with np.load(out/"exact_half_feature_covariances.npz") as covs, np.load(out/"paired_distance_samples.npz") as samples:
        assert len(covs.files) == len(samples.files) == len(records)
        for rec in records:
            design = next(d for d in frozen["designs"] if d["name"] == rec["design"])
            point = next(p for p in frozen["points"] if (p["panel"], p["point"]) == (rec["panel"], rec["point"]))
            cov, draws = covs[rec["covariance_archive_key"]], samples[rec["samples_archive_key"]]
            checks["max_covariance_asymmetry"] = max(checks["max_covariance_asymmetry"], float(np.max(abs(cov-cov.transpose(0, 2, 1)))))
            checks["minimum_covariance_eigenvalue"] = min(checks["minimum_covariance_eigenvalue"], float(np.linalg.eigvalsh(cov).min()))
            f, _ = experiment.features(*experiment.local_model(point["theta"], point["x"]), design["sets"])
            delta = f[:16]-f[16:]
            variance = experiment.distance_variance(delta, cov, np.array(design["metric"]))
            checks["max_variance_recomputed_error"] = max(checks["max_variance_recomputed_error"], float(np.max(abs(variance-rec["exact_pair_variances"]))))
            checks["max_sample_mean_recomputed_error"] = max(checks["max_sample_mean_recomputed_error"], float(np.max(abs(draws.mean(axis=0)-rec["empirical_pair_means"]))))
            checks["max_sample_variance_recomputed_error"] = max(checks["max_sample_variance_recomputed_error"], float(np.max(abs(draws.var(axis=0, ddof=1)-rec["empirical_pair_variances"]))))
            w = .1*np.array(rec["original_angle_jacobian"])/np.sqrt(2*variance)[:, None]
            checks["max_whitening_recomputed_error"] = max(checks["max_whitening_recomputed_error"], float(np.max(abs(w-rec["whitened_jacobian"]))))
            assert rec["rank"] == (3 if design["products"] == 0 else 4)
            assert rec["actual_shots_all_pairs_one_state"] == 32*rec["budget"]
            assert rec["actual_shots_all_pairs_two_states"] == 64*rec["budget"]
            assert rec["whole_space_weakest"] == (rec["whitened_singular_values"][-1] if rec["rank"] == 4 else 0)
    assert checks["minimum_covariance_eigenvalue"] > -1e-12
    assert max(v for k, v in checks.items() if k.startswith("max_")) < 1e-12
    assert (out/"minimal_product_results.tex").read_text() == experiment.latex(summary)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(experiment.HERE), pattern="test_experiment.py")
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    assert result.wasSuccessful()
    report = dict(passed=True, phase="post-run audit; no design modification", tests_run=result.testsRun,
                  unit_test_output=stream.getvalue(), records=len(records), scientific_checks=checks,
                  result_manifest_sha256=experiment.sha(out/"RESULT_MANIFEST.json"),
                  frozen_manifest_sha256=experiment.sha(experiment.HERE/"FROZEN_MANIFEST.json"))
    experiment.write_json(experiment.HERE/"AUDIT_RESULTS.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
