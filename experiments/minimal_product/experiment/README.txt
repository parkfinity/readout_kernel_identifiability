Minimal-product finite-shot comparison
=====================================

Dependency: Python 3.9+ and NumPy (no quantum SDK, network or server required).
Read PROTOCOL.txt for the prespecified scope and all comparison choices.

The delivered frozen_inputs.json and FROZEN_MANIFEST.json are authoritative.
Run from this directory:
  python3 -m unittest -v test_experiment
  python3 run.py run
  python3 audit_results.py
Optional fresh-directory replay and exact local comparison:
  python3 replay.py
The run refuses to overwrite results. To replay, copy the experiment directory
to a new empty working location WITHOUT its results directory, leaving all
frozen files unchanged, then run the two commands above. Floating-point output
may vary slightly across NumPy/BLAS versions; compare scientific arrays with
rtol=1e-9, atol=1e-11, not wall-clock time, paths or archive byte hashes.

Maintainer-only freeze (already performed before this run):
  python3 run.py freeze --source /path/to/readout_kernel_identifiability
Freeze checks original published hashes and extracts all 15 Hubregtsen test
points. It does not evaluate designs or select points. The default --source is
the sibling public-release directory in the manuscript workspace.

Outputs:
  results/summary.json: every panel/arm/estimator/budget summary and comparison.
  results/records.json: all 480 records, Jacobians, variances, singular values,
    shot accounting, Monte Carlo means/variances and calibration checks.
  results/source_checks.json: all-point checks against actual source gate IR.
  results/exact_half_feature_covariances.npz: exact full feature covariances.
  results/paired_distance_samples.npz: all 256-replicate, 16-pair raw samples.
  results/minimal_product_results.tex: generated manuscript-ready subsection.
  results/RESULT_MANIFEST.json: output file hashes.
  AUDIT_RESULTS.json: post-run unit-test log and saved-array consistency audit.
  REPLAY_VALIDATION.json: fresh-directory local replay evidence.

The source program is embedded in frozen_inputs.json. Replaying the delivered
freeze does not require the original public-release directory or an absolute
machine-specific source path. Only the maintainer freeze operation reads that
directory. audit_results.py was added after the run as a read-only consistency
check of frozen inputs, test outcomes, output hashes and saved arrays; it is not
part of the pre-execution experimental design. It writes only AUDIT_RESULTS.json.
replay.py is likewise a post-run verification helper. It copies only the frozen
files to a new temporary directory, reruns tests and the full experiment there,
and checks records, summaries, covariances and paired samples against the saved
reference. It retains the temporary directory and records only its basename
in the report, not the local account-specific parent path.

Archive keys in each record identify the corresponding NPZ arrays. Covariances
have shape (32 inputs, number of features, number of features); samples have
shape (256 replicates, 16 input pairs). Four local features precede products in
the explicit frozen order. Feature/source calculations are standalone NumPy.
The dense-source checker propagates the actual original compiler program; it
does not assume the analytic product model. Exact covariance formula and
split-distance variance follow the existing public core/product controls,
with an independently coded input-major layout and enumeration-based tests.

All points and original sampling seeds are reused from a completed historical
study, so "frozen" does not mean new preregistration. Metrics, estimator arms,
Q=I4, budgets and this code were frozen before the new experiment ran. Rank
proof is separate; these are model diagnostics, not detection/QPU evidence.
