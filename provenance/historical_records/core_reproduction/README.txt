REVIEWER REPRODUCTION PACKAGE — version 1
Readout-Dependent Parameter Identifiability of Quantum Kernels
Reference: archived manuscript tables. Prepared 2026-10-04.

START HERE
==========
This package reconstructs the central dimension/certificate census and the
strong-baseline resource audits. It contains the scientific source, frozen
circuit inputs, upstream source snapshots, expected certificates, small server
result archives, an English claim map, and a non-destructive runner.

No QPU, cloud account, API key, image dataset, or training service is needed for
the core experiment. After Python dependencies are installed, the reproduction
commands work offline. Nothing is uploaded. This is a reviewer distribution,
not a claim that the code has already been published in a public repository.

IMPORTANT SCOPE
The main 54-setting census is 24 pilot settings plus 30 disjoint extension
settings, not 54 independently sampled tasks. The scaling study contains 12
size/readout cases, some overlapping that census. Rank means generic
differential kernel rank, not necessarily a global coordinate parametrization.
Python exact/interval checks do not replace review of the written proofs.

QUICK START (Linux/macOS; Windows users: use WSL)
================================================
Unzip the package, then enter the directory containing THIS README and run.sh.
Use a separate environment with Python 3.9–3.11. These frozen NumPy/SciPy
versions are not intended for Python 3.12+. Python 3.8 is historically supported
by the original core scripts, but is not the recommended new installation.
For the existing Python 3.8 server environment, requirements.txt selects the
legacy pbr dependency and the runner retains the original math.gcd shim.
The new integrated release was locally tested on Python 3.9, not freshly on
Python 3.8; keep the emitted environment log when reproducing on that server.
Legacy dependency metadata: https://pypi.org/project/pbr/6.1.1/

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    bash run.sh integrity
    bash run.sh smoke
    bash run.sh audit
    bash run.sh full --workers 2

If python3 is the wrong interpreter, use e.g. python3.11 for creating the venv.
Paste only command lines: do not paste Markdown fences, prompt characters, or
the text "```bash" into a terminal. To select an existing interpreter:

    PY=/absolute/path/to/python bash run.sh full --workers 2

WHAT EACH COMMAND DOES
======================
integrity
  Standard-library SHA256 verification of delivered files. No scientific claim
  follows from matching hashes alone. No environment installation is required.

smoke
  Unit tests of the exact/interval, Gram, scaling, normalizer, and survey code;
  exact endpoint-minor sanity checks. This is not a rerun of the 54-setting census.

audit
  Smoke plus recomputation of the 24 stored pilot certificates at higher interval
  precision, source/IR checks, nonlinear exact identities and 1,000 source-pair
  checks, QFI counterexample, bounded analytic checks, and archival CSV
  reaggregation. It generates census_54.csv from the stored/fresh-audited inputs.
  It does NOT recompute all 30 Gram certificates or the 12 scaling cases.

full
  Smoke + fresh 24-setting discovery and replay + source checks + nonlinear and
  analytic checks + fresh 30-setting Fourier–Gram computation + fresh Kyriienko
  n=2,4,6,8 certificates (three readouts) + comparisons to stored certificates.
  Then fresh 102-distinct-setting resource survey and 12-setting global
  normalization validation. Writes a combined census_54.csv.
  "Full" here means the core identifiable-dimension/resource reproduction,
  NOT every historical appendix training experiment or a hardware experiment.

resources
  Runs only the fresh resource survey and global-normalization experiment and
  compares the structural outcomes with the supplied original server results.

timing-smoke
  Small test of the historical seven-arm training/timing code on frozen inputs.
  Separate from the main full command. Not a performance estimate.

diagnostics
  Recomputes the sampled direct-summand rank diagnostic (with constant mode)
  at n=2,4,5,6,7,8,10,12 and seeds 1,7. This is a floating sampled-rank check,
  NOT an exact dimension certificate; condition failure is a reported result.

timing-full
  Explicit opt-in to the historical 18,900 cold fits + 3,600 reuse fits.
  This can be costly and is NEVER run automatically by full. Timings are
  machine-dependent and must not be expected to equal the original wall times.
  Run alone with stable load, and report hardware, package versions and threads.
  This benchmark predates the stronger global normalization; it is not evidence
  of a speedup over that later baseline or of QPU savings.

timing-archive
  Reaggregates saved pairs and timing arithmetic from the optional large
  companion. This is NOT simulator or timing replay. Place the companion next
  to the extracted package, then run:

    bash run.sh timing-archive --archive ../quantum_reviewer_timing_evidence_v1.zip

  Its required SHA256 is:
  f8185e3287a9bbb602062f64cb483892c586b6433d981be85de6063487a5e871

OUTPUT, PROGRESS AND FAILURE HANDLING
====================================
Each command writes runs/<mode>_<UTC>/, including:
  verdict.json             aggregate pass/fail and explicit mode
  stages.json              stage exits and elapsed times
  environment.json         Python/package versions and platform
  logs/                    complete stdout/stderr, including test counts
  work/                    private copy of scientific code and generated output
  census_54.csv            audit/full only; per-setting lower/upper ranks
  core_comparison_verdict.json  full only; detailed certificate comparisons
  resource_comparison.json     resources/full only

To additionally check that the emitted census matches both archived rank tables:

    python check_manuscript_tables.py runs/<actual-run-directory>/census_54.csv --output table_check.json

This checks presentation consistency, not certificate validity.

The runner prints [RUN], [WORKING] every 15 seconds during long stages, and
[PASS]. [WORKING] is only liveness, not a scientific success verdict. A shell
prompt returning after PASS means the command has finished. A failure stops
the workflow with a nonzero exit and identifies its log. Never use python -O
or PYTHONOPTIMIZE because original scientific scripts use assertions.

Files supplied in core/ and evidence/ are never overwritten: each run copies
them into its work directory. Different top-level runs have separate working
copies; do not run the same underlying script concurrently in one directory.
Default output can be changed with --out /path/to/reviewer-results.

Send verdict.json, stages.json, environment.json, comparison files and logs
when reporting an issue. Retain work/ for inspection of exact certificates.
No zip shell utility is necessary; Python's zipfile module can pack a run:

    python -m zipfile -c reviewer_return.zip runs/<actual-run-directory>

EXPECTED SCIENTIFIC OUTCOMES
============================
24-setting pilot: 162 symbols, 132 ACTIVE_CERTIFIED, 30 EXACT_INACTIVE;
  24 closed ranks, 22 nonzero-minor witnesses and two rank-zero settings.
30-setting Gram closure: 252 symbols, 195 ACTIVE, 57 INACTIVE;
  30 closed ranks, no unresolved rank interval. Together: 54 closed settings.
Hubregtsen nonlinear example: exact generic rank 2, constant linear-factor
  rank 4; all four retained coordinates active. No extra gate/shot saving.
Kyriienko local XYZ ranks at n=2,4,6,8: 4,10,16,22.
  Weight<=2 ranks: 0,10,16,22. Local Z ranks: 4,8,12,16.
  The n=2 complete-readout exception is essential. Equality for all even n
  is NOT established by finite computations.
Survey: 120 nominal slots, 18 duplicates, 102 distinct; four additional-symbol
  slots but zero additional fused-operation/trainable-angle-task reductions.
Global normalization: 12 settings; no extra trainable-angle-task reduction
  from transport after global normalization. The representative tasks are
  3->3 (2 qubits) and 7->7 (4 qubits), not physical native-gate/shot counts.

Symbolic budget completions and floating diagnostics can vary by machine.
The original core comparator separates strict certificates from such warnings.
Read any PASS_WITH_WARNINGS result; do not discard its warning section.
The runner never replaces a failed identity by a tolerance-based certificate.

COMPUTE AND SOFTWARE
====================
CPU-only. Plan for 8 GB available RAM and up to about 30 minutes for an initial
core run; this is a planning allowance, not a measured guarantee on your machine.
Use --workers 1 on memory-constrained machines, --workers 4 or 8 only with room
for concurrent exact symbolic computations. Prior 8-worker runs used about
6.3 GB aggregate memory. New delivery-specific timings are in VALIDATION.txt.
Numerical libraries are single-threaded by the runner to reduce oversubscription.

The pinned upstream circuit subset and its original Apache-2.0 license/CITATION
are included in vendor directories. Full installation of sQUlearn is not needed;
the documented source host instantiates the frozen subset through Qiskit.
requirements.txt pins the installation environment. The original core comparator
also records historical Python-version differences; matching Python microversions
is not required to check exact outcomes. See NOTICE.txt before public release.

LAYOUT / PROVENANCE
===================
core/ contains the original reproduction_bundle_v1, plus byte-preserved source
  modules for the pilot, nonlinear, resource and optional timing experiments.
  BUNDLE_MANIFEST.json there covers the original subset; MANIFEST.json at this
  package root covers the complete delivered package.
evidence/ contains original small server archives and reported summaries.
manuscript_reference/ contains archived TeX fragments/BibTeX for locating claims.
  It is NOT a compilable manuscript upload (figures/class files are separate).
CLAIM_MAP.txt links claims to commands, inputs, outputs, and limitations.

Archived text fragments in core are retained solely because original
certificate provenance hashes refer to them. The article text controls
scientific claims; the old fragments are not an alternative current manuscript.
Historical environment records may contain author-machine paths; no account,
service credential, private key or login is needed to execute this package.

EXCLUSIONS
==========
Raw industrial images/embeddings, original QPU jobs, and every historical
training run are NOT included in the main ZIP. The supplied legacy CSV route
reaggregates saved results, rather than retraining them. The optional timing
workflow starts from frozen preprocessed WDBC/Ionosphere/Auto MPG inputs; it
does not independently reconstruct the earlier raw-data preprocessing.
See CLAIM_MAP.txt for explicit distinctions. Do not describe this package as
full end-to-end reproduction of every appendix result or as a formal proof.
