"""Fresh-directory same-code replay; no original-release path required."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import run as experiment


def main():
    experiment.load_frozen()
    destination = Path(tempfile.mkdtemp(prefix="minimal_product_replay_"))
    names = list(json.loads((experiment.HERE/"FROZEN_MANIFEST.json").read_text()))
    for name in names + ["FROZEN_MANIFEST.json"]:
        shutil.copy2(experiment.HERE/name, destination/name)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    tests = subprocess.run([sys.executable, "-m", "unittest", "-v", "test_experiment"],
                           cwd=destination, env=env, capture_output=True, text=True, check=True)
    result = subprocess.run([sys.executable, "run.py", "run"], cwd=destination,
                            env=env, capture_output=True, text=True, check=True)
    reference, replay = experiment.HERE/"results", destination/"results"
    for name in ["records.json", "source_checks.json", "minimal_product_results.tex"]:
        assert (reference/name).read_bytes() == (replay/name).read_bytes(), name
    maxima = {}
    for name in ["exact_half_feature_covariances.npz", "paired_distance_samples.npz"]:
        maximum = 0.0
        with np.load(reference/name) as original, np.load(replay/name) as repeated:
            assert original.files == repeated.files
            for key in original.files:
                np.testing.assert_allclose(original[key], repeated[key], rtol=1e-9, atol=1e-11)
                maximum = max(maximum, float(np.max(abs(original[key]-repeated[key]))))
        maxima[name] = maximum
    # Runtime and process path are administrative metadata, not scientific output.
    original = json.loads((reference/"summary.json").read_text())
    repeated = json.loads((replay/"summary.json").read_text())
    for value in [original, repeated]:
        value.pop("elapsed_seconds")
    assert original == repeated
    report = dict(passed=True, scope="Fresh-directory same-code local replay, not cross-platform/server/QPU validation",
                  temporary_replay_directory=destination.name, frozen_manifest_sha256=experiment.sha(experiment.HERE/"FROZEN_MANIFEST.json"),
                  exact_record_file_match=True, scientific_summary_match=True,
                  maximum_absolute_array_differences=maxima, tests_stdout=tests.stdout,
                  tests_stderr=tests.stderr, replay_stdout=result.stdout, replay_stderr=result.stderr)
    experiment.write_json(experiment.HERE/"REPLAY_VALIDATION.json", report)
    print(json.dumps({k: v for k, v in report.items() if "stdout" not in k and "stderr" not in k}, indent=2))


if __name__ == "__main__":
    main()
