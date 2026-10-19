RECOVERY VERIFICATION SUPPLEMENT
Readout Dependent Parameter Identifiability of Quantum Kernels
19 October 2026

This ZIP supplements, and does not replace, the existing public package:
https://github.com/parkfinity/readout_kernel_identifiability
Baseline commit: d53cdfe9ee254c0d687ebec28f13584218f5ff04
The new exact and finite-shot work is self-contained. The 54-setting evidence
index additionally requires the baseline package at the pinned revision.
Nothing in this supplement claims a QPU run, formal proof-assistant check,
global unique recovery, or guaranteed predictive/shot-efficiency improvement.

NEW EXACT RESULT
The original four local means have rank <=3 under every fixed quadratic metric.
Appending f0*f2 and the explicit SPD metric M1/7 gives generic rank4, certified
by determinant -9/307328 at theta=(pi/6)^4. One scalar addition is minimal if
the original means are retained and any fixed SPD metric is allowed.
Verification includes all 17 exact tests and counterexamples to stronger claims.

NEW MATCHED FINITE-SHOT STUDY
Zero, one, two and six products; fixed unit-trace metrics; 15 reused original
evaluation points; 16 disjoint pairs; 4 shot budgets; 2 estimators; 480 records.
Full covariance and original-angle target Q=I4. All 1/2/6-product points have
numerical rank4. One-product minimality does not imply better conditioning.
The reference was calculated with Python 3.9.6, NumPy 1.26.4 and SymPy 1.14.0.
It was fully replayed in a fresh directory; the same-environment array
differences were zero. This is not a server/cross-platform replay.

START HERE
Use Python 3.9 or newer in a separate environment; install requirements.txt.
From this extracted supplement root:
  python3 verify_supplement.py

This checks every manifest hash and runs the 17 exact + 8 experiment tests.
It does not change the frozen scientific inputs or reference results.
To independently replay all 480 records into a fresh temporary directory:
  cd experiment
  python3 replay.py

The replay writes a new REPLAY_VALIDATION.json audit report. Run the package
integrity check before, not after, that deliberate report update, or use a
second fresh extraction. Do not rerun the maintainer 'freeze' operation.
experiment/run.py refuses to overwrite an existing results/ directory.

54-SETTING INDEX (BASELINE DEPENDENCY)
Keep this supplement next to a checkout named readout_kernel_identifiability,
or give its actual location explicitly:
  python3 census_mapping/verify_census.py --package /path/to/readout_kernel_identifiability
  python3 census_mapping/verify_census.py --package /path/to/readout_kernel_identifiability --replay-modular

To run the six index acceptance/rejection tests, set READOUT_KERNEL_PACKAGE
to the checkout path and run:
  python3 -m unittest discover -s census_mapping -v

Only the two census table sources are copied under manuscript/ for the index.
They are not a complete manuscript. See census_mapping/README.txt for the
24 product-angle versus 30 fixed-frequency routes. All 54 table ranks and 114
manifest files matched. All 60 modular records were replayed (54 nonempty,
6 empty rank-zero minors). Symbolic upper bounds were NOT regenerated in this
update. The baseline's core-full command is the fresh full-generation route.

FILES AND RIGHTS
verification_math/: exact algebra, tests, expected certificate, proof fragment.
experiment/: frozen protocol/inputs/code, every result, raw paired draws,
              post-run audits and independent read-only review record.
census_mapping/: complete 54-row evidence index, checker and replay report.
MANIFEST.json: SHA-256 hashes of every distributed file except the manifest.
LICENSE: MIT license copied from the public project for the project code.
No vendored external library is copied into this addendum; its optional
baseline dependency retains its own third-party notices and licenses.

AVAILABILITY BEFORE SUBMISSION
The manuscript explicitly distinguishes this addendum from the old public
commit. Upload this supplementary ZIP with the submission, or add it to the
public repository and then update the manuscript to cite the new commit.
The assistant has not pushed or published these new files.
