# Readout Dependent Parameter Identifiability of Quantum Kernels

Reproducibility materials for the manuscript by Hyong Keun Park, Jahoon Jeong,
Joonkyo Shim, and Hyunsoo Yoon. Correspondence: hs.yoon@yonsei.ac.kr.

The public folders use descriptive names, not internal development-version
labels. Scientific values, array payloads and exact certificates are preserved. Public metadata
was sanitized as documented in provenance/PUBLIC_METADATA_CHANGES.json.
A separate numerical follow-up adds the missing Kyriienko (n,k)=(5,3) setting. The
Kyriienko verifier includes a Python integer-serialization compatibility fix;
its exact change and the pre-fix source hashes are recorded in provenance.

## Quick start

Use Python 3.9, 3.10 or 3.11 on Linux/macOS, or WSL on Windows. The pinned
requirements are not intended for Python 3.12+. No QPU account, API key, GPU
or image download is required for the main workflows.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py integrity
python reproduce.py smoke
python reproduce.py core-audit
```

Use another installed Python 3.9-3.11 interpreter if appropriate. Do not use
`python -O` or `PYTHONOPTIMIZE`; scientific assertions must remain enabled.
Dependency installation requires internet access. The calculations run offline.
The reference server runs used Python 3.8.10. The public quick start above uses
Python 3.9-3.11 and is not a claim that the historical server used that version.

To replay the 12 stored Kyriienko determinant certificates directly:

```bash
python reproduce.py scaling-verify
```

This command restores the historical reference layout before invoking
`kyriienko_scaling.py --verify`. Running that script directly against the renamed
public folders omits moved provenance files, so use the root entry point.
This replay checks stored Jacobians and recomputes exact determinants; it is
not a fresh generation of all circuits. The verifier disables the integer-string
digit limit for these trusted local certificates. Do not use it to parse
untrusted network input.

## Workflow selection

| Command | Calculation | Limit |
|---|---|---|
| `python reproduce.py core-audit` | Replay saved pilot certificates, source checks and census aggregation | Not a fresh discovery of every rank |
| `python reproduce.py core-full --workers 2` | 24-setting pilot, 30-setting Gram audit, scaling cases and resource surveys | Not every historical learning experiment |
| `python reproduce.py recovery` | Metric/readout survey, exact certificates and correlation-lift controls | Floating diagnostics are distinct from exact proofs |
| `python reproduce.py shots-pilot` | Small frozen-plan joint-shot diagnostic | Not the full reported study |
| `python reproduce.py shots-full` | Three panels, four settings, five held-out points, four budgets and 256 replicates | Ideal model, not QPU or classification results |
| `python reproduce.py product` | Product-feature covariance and paired simulation controls | Uses archived direct estimates for comparison |
| `python reproduce.py ad-summary` | Category-level AD reaggregation and bootstrap intervals | Not raw-score replay, image inference or training |
| `python reproduce.py kyriienko-weight3` | Fresh (5,3) scan, five seeds, dense and pinned-source checks, saved-array comparison | Numerical rank 13, not an exact certificate |

The smoke workflow contains 97 unit tests plus endpoint and frozen-plan checks.
Integrity checking detects changed files; it is not scientific verification.
See `VALIDATION.txt` for which workflows were actually retested after packaging.

## Public directory layout

```text
experiments/
  metric_recovery/     Readout and metric recovery
  finite_shot/         Frozen-plan finite-shot sensitivity
  product_control/    Classical product-feature control
  shot_reference/     Original paired direct-estimator samples
  kyriienko_weight3/  Five-seed (5,3) numerical follow-up and full sampled arrays
core_reproduction/    Dimension certificates, scaling and resource audits
reference_results/    Reported numerical evidence
tools/                Supporting analysis
provenance/           Original path map, historical text and validation records
reproduce.py          Supported public entry point
requirements.txt      Reproduction environment
```

## Public paths and provenance

`provenance/PATH_MAP.json` maps the neutral runtime layout to public files.
The launcher reconstructs this layout inside `runs/<workflow>_<UTC>/runtime/`;
no author's home directory or external manuscript checkout is needed.
Personal host paths and submission-history wording were removed from metadata.
Directory aliases for manuscript evidence are now journal-neutral.

`provenance/PUBLIC_METADATA_CHANGES.json` records exact before/after hashes.
Scientific numerical values, NPZ arrays and compressed exact Jacobians were
not changed. Historical generator hashes remain historical, not claims of a
new run. Current manifests cover sanitized files. Fresh-run comparisons allow
only explicitly pinned metadata/source-hash transitions; unknown changes fail.
The Python integer-serialization compatibility fix is separately documented.
Sanitized archived snapshots need not match their recorded pre-cleanup hashes.

Use the root entry point. Runs save logs and `verdict.json` in a fresh `runs/`
folder and do not overwrite supplied references. The `kyriienko-weight3`
workflow uses the public layout directly.

## Scientific scope

- The census contains 24 pilot and 30 disjoint extension settings, not 54
  independently sampled tasks. Generic differential rank is not automatically
  a global minimal coordinate parametrization.
- Exact dimension, coordinate activity and executable cost are separate claims.
  Zero incremental fused-angle-task findings are retained.
- The (5,3) follow-up closes the missing numerical scan entry, not an exact
  odd-qubit certificate. It does not change the separate 54-setting census.
- For the specified Hubregtsen recovery example, isotropic, dense-metric and
  lifted-feature ranks are 2, 3 and 4. The ideal lifted features are also
  obtainable classically from local features in this example.
- Finite-shot sensitivity does not establish classification accuracy, detection
  probability, universal metric-selection superiority or hardware savings.
- The product-estimator control assumes a factorized ideal Z distribution.
- AD is a diagnostic/negative learning control, not a test of rank recovery.

Raw industrial images, embeddings, original QPU jobs, credentials and the large
historical paired-timing archive are excluded. Optional timing code starts from
frozen preprocessed UCI inputs, not a fresh raw-data pipeline. The standalone
historical timing archive is not reconstructed by this repository. Current and
historical TeX fragments under provenance locate claims; they are not compilable
manuscript uploads or alternative current scientific claims.

## Licensing, citation and upload

The authors' original software is licensed under MIT; see `LICENSE` and
`LICENSE_STATUS.txt`. Vendored sQUlearn code retains Apache-2.0. See
`THIRD_PARTY_NOTICES.txt` for third-party code and dataset terms. External library versions,
licenses, DOI identifiers and original evidence timestamps are intentionally
preserved. Do not add raw third-party datasets or service credentials.

Repository: https://github.com/parkfinity/readout_kernel_identifiability

Upload the contents of this folder to the repository root (do not nest a second
copy of the repository folder). Keep `.gitattributes` to preserve file hashes.
Do not upload `runs/`, virtual environments, or credentials. After uploading,
download a fresh ZIP and repeat integrity, smoke, core-audit, and
`python reproduce.py kyriienko-weight3`. Enable public
visibility and test the exact manuscript URL while logged out before submission.
The repository was still private when this release was prepared.

Cite the manuscript and the repository commit used for reproduction. A Zenodo
DOI can be added after a public GitHub release has been archived; no DOI has
been assigned by this packaging operation.
