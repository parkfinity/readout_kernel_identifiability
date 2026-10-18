# Readout Dependent Parameter Identifiability of Quantum Kernels

Reproducibility materials for the October 7, 2026 manuscript by Hyong Keun Park,
Jahoon Jeong, Joonkyo Shim, and Hyunsoo Yoon.

This release brings together the earlier dimension/resource reproduction package
and the subsequent readout-metric, finite-shot, and classical product-feature
controls. It is a research artifact, not a general-purpose quantum compiler.
See [UPLOAD_KO.txt](UPLOAD_KO.txt) for upload instructions.

## Quick start

Use Python 3.9, 3.10 or 3.11 on Linux/macOS (Windows: WSL). The pinned packages
are not intended for Python 3.12+. A CPU is sufficient. The core workflows need
no IBM account, API key, QPU, GPU, or image download. Package installation needs
internet access; the documented calculations then run offline.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py integrity
python reproduce.py smoke
```

Use another installed 3.9-3.11 interpreter in the first command if necessary.
Do not use `python -O` or `PYTHONOPTIMIZE`: scientific assertions must remain on.
Do not run the historical module scripts directly unless you intend to manage
their output paths yourself. The wrapper creates separate working copies.

## Select the calculation to reproduce

| Command | What is freshly computed | Important limit |
|---|---|---|
| `python reproduce.py core-audit` | Earlier exact/interval replay, selected identities, source checks and archived census aggregation | Not a fresh discovery of all 54 ranks |
| `python reproduce.py core-full --workers 2` | 24-setting pilot, 30-setting Gram audit, 12 scaling cases, resource surveys | Not a replay of every historical learning experiment |
| `python reproduce.py recovery` | Metric/readout survey, exact controls, certificate checks, strong controls and correlation lift | Numerical rank diagnostics are distinct from exact certificates |
| `python reproduce.py shots-pilot` | Small frozen-plan ideal joint-shot run and analysis | A smoke-scale diagnostic, not the full reported study |
| `python reproduce.py shots-full` | Three panels, four settings and five held-out points per panel/setting, four budgets, 256 replicates | Ideal-model design, not a hardware or predictive-accuracy result |
| `python reproduce.py product` | Same-outcome product-feature simulation and covariance checks, compared with archived direct estimates | Historical direct samples are supplied; this is not a new QPU run |
| `python reproduce.py ad-summary` | Category aggregation, category-bootstrap intervals, matched-cohort contrast | Starts from saved category means, NOT images or saved per-image scores |

`smoke` runs the earlier core tests plus the recovery, shot and product tests.
It is not a full experiment run. Integrity checking only detects changed bytes.

Outputs are placed under `runs/<mode>_<UTC>/`: `verdict.json`, `stages.json`,
`environment.json`, stage logs, and private working copies. `[WORKING]` is a
liveness message; only a final PASS/nonzero exit indicates the selected workflow
completed/failed. See [VALIDATION.txt](VALIDATION.txt) for the checks actually run
on this integrated delivery. Historical validation is retained separately.

## Claims and expected results

- The dimension census has 24 pilot and 30 disjoint extension settings, not 54
  statistically independent tasks. Generic differential rank is not automatically
  a global minimal coordinate chart.
- The nonlinear Hubregtsen example has generic rank 2 but constant linear-factor
  rank 4, with all four retained coordinates active.
- Kyriienko local XYZ ranks at n=2,4,6,8 are 4,10,16,22; weight<=2 ranks are
  0,10,16,22. Finite scaling checks do not by themselves prove all-n equality.
- The 102-distinct-setting resource survey and global normalization audits must
  retain their zero additional fused-angle-task results. Lower rank does not
  imply an unconditional QPU or shot saving.
- The recovery control distinguishes isotropic rank 2, dense-metric rank 3,
  and lifted-feature rank 4 for the specified Hubregtsen example. Here the lifted
  ideal features can be built classically from the local features; this is not
  evidence of an unavoidable need for additional quantum measurements.
- Shot sensitivity is not detection probability or classification accuracy.
  Noise-aware selection is not uniformly better at every held-out point.
- The product estimator control concerns a factorized ideal Z distribution.
  Its variance advantage is not established for arbitrary correlated/noisy states.
- AD results are negative/diagnostic learning controls, not a test of the
  rank-recovery method.

## Contents and provenance

- `core_reproduction/`: byte-preserved earlier reviewer package, including its
  internal manifest, source snapshots, reference evidence and historical logs.
  Its archived descriptions and distribution notices provide context,
  not the scope statement for this integrated repository.
- `experiments/`: later modules with their original plans, scientific source and
  input manifests. Original development README files are retained for provenance.
- `reference_results/`: recovery results, shot server JSON, product results and AD
  category summary. These are reference evidence, not results newly generated
  merely by cloning the repository.
- `manuscript_reference/`: current TeX text to locate claims; NOT a standalone
  compilable manuscript (figures, class and bibliography style are not included).
- `MANIFEST.json`: immutable executable/input/reference files. Root documentation,
  current manuscript text, and generated `runs/` are deliberately outside the
  integrity scope so maintainers can update repository metadata.

The three new literature citations locate our scope relative to Fourier
expressivity, generalized parameter-shift rules, and finite-sampling measurement
allocation. No newly cited method is claimed to have been implemented here.

## Exclusions and access limits

This repository does not contain raw industrial images, private/company data,
IBM credentials, original QPU jobs, or the 268 MB historical paired-timing archive.
The earlier runner includes optional timing commands, but archive reaggregation
requires the separate original archive. Do not claim that the repository alone
reproduces that archive, every appendix training run, or raw-data preprocessing.
The frozen UCI arrays support optional timing from preprocessed inputs only.
Table-level AD reaggregation does not repeat the earlier 270-score-array audit.

Original provenance records retain some historical local filesystem paths.
Those paths are metadata, not required locations or network access. Run through
the root entry point; private maintainer-only packaging scripts are not workflows.

## Third-party material and licensing

See [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and
[LICENSE_STATUS.txt](LICENSE_STATUS.txt). Preserve all vendor notices and data
attribution. A code license never relicenses third-party benchmark data.

## Citation and contact

Please cite the manuscript once its arXiv or journal identifier is available.
Until then, use the title and authors above and identify the repository commit
used in a reproduction. No arXiv number, DOI, acceptance, or public upload is
claimed by this prepared release. Correspondence: hs.yoon@yonsei.ac.kr.
