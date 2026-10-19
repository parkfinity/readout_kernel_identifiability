54-SETTING EVIDENCE INDEX — revision supplement, 2026-10-19

Scope
-----
This index maps the current manuscript's 24 pilot and 30 disjoint extension
settings to actual files in the public reproduction package. No public files
are modified. The commit supplied for manuscript citation is:
  d53cdfe9ee254c0d687ebec28f13584218f5ff04
  https://github.com/parkfinity/readout_kernel_identifiability
The checker verifies local release-manifest bytes, not remote Git state.
This index is new revision material; it is not claimed to exist at that commit.

Files
-----
census_54_mapping.csv   Compact one-row-per-setting map, including exact rank,
                       source/artifact hashes, proof fields and commands.
census_54_mapping.json  The same 54 settings with detailed minor records,
                       rational determinants when stored, denominator checks,
                       upper-bound route, source hashes and scope limitations.
verify_census.py        Read-only default consistency checker; optional fresh
                       modular lower-certificate replay.
test_verify_census.py   Six acceptance/rejection tests without source writes.
mapping_verdict.json   Result of the default mapping check.
modular_replay_verdict.json
                       Fresh modular coefficient/Jacobian computation and
                       stored-minor replay for both primes in all 30 settings.

Interpretation
--------------
24 pilot settings: ChebyshevRx and ParamZFeatureMap, n=2,4; layers=1,2;
local XYZ, local Z and complete Pauli weight <=2 readouts. All contain a
trainable-data product angle. Exact formal coefficient cancellation provides
constant-null-space upper bounds; feasible interval Jacobian minors provide
lower bounds. There are 22 nonempty minors and two exact-zero kernels.
The fixed-frequency Gram theorem is not used for this product-angle cohort.

30 extension settings: YZ-CX and Hubregtsen, n=2,4; layers=1,2; Kyriienko,
n=2,4 with one encoding/variational layer; the same three readouts. All
pass the fixed-frequency/no-constant-angle contract. Distinct data-coordinate
phases have open images, so the finite independent trigonometric basis
supports the coefficient-Gram rank correspondence.

The index uses the later Fourier-Gram result as its uniform 30-setting route,
not the earlier extension's partially open rank records. Its upper bound is
the minimum of the original symbol count, exact two-copy invariance-field
bound U1, and any completed original/reduced symbolic-Gram span bound U2.
upper_bound.attained_by gives every stored bound attaining the reported rank.
The nonlinear Hubregtsen n4/l1/local-Z rank 2 is closed by U2, not its U1
bound. The three formerly open Kyriienko settings are closed by U1 including
virtual Euler fields. A numerical J(theta*) v(theta*)=0 check alone is not
the global upper proof; the exact invariance identities are essential.

There are 60 modular records, but only 54 nonempty positive-rank minors
(27 positive-rank settings, two primes each). Three rank-zero settings use
the empty determinant convention det([])=1. Seven optional exact-rational
records include six nonempty minors and one empty minor. Empty minors do
not prove a positive rank. All 54 table ranks match the stored certificates;
327 source coordinates are active and 87 inactive across the two cohorts.

A nonzero determinant modulo p implies a nonzero rational determinant only
when reduction is defined on the coefficient ring. The checker tests primality,
Pythagorean rational-circle identities and invertibility modulo each prime
of 2, phase denominators, rational metric denominators and half-angle
coefficient denominators. These generate the denominators of the released
Laurent propagation and rational Gram Jacobian. The replay also bounds
integer contractions below signed int64 capacity and integer-valued float64
binning below 2^53 for these exact settings. These bounded implementation
checks do not establish correctness for unrestricted circuits.

How to verify
-------------
Run from this census_mapping directory:

  python3 verify_census.py

This checks membership (24+30, no overlap), all 54 manuscript table ranks,
upper/lower equality, feasible stored pilot input coordinates, stored minor
conditions, rational denominators, and 114 input/certificate/source files
against the package's release manifest. It also compares the CSV/JSON map
against a fresh extraction. This is NOT a certificate replay: stored
identities and interval endpoints are not recomputed in default mode.

To locate another copy of the public package explicitly:

  python3 verify_census.py --package /absolute/path/readout_kernel_identifiability

If the manuscript is not the sibling manuscript/ directory, additionally pass:

  --manuscript /absolute/path/manuscript

To run the six checker acceptance/rejection tests:

  python3 -B -m unittest -v test_verify_census

To regenerate the index (only this supplement's CSV/JSON files are written):

  python3 verify_census.py --build

To replay both modular records for all 30 extension settings:

  python3 verify_census.py --replay-modular --output modular_replay_verdict.json

To replay one named setting:

  python3 verify_census.py --replay-modular --setting hubregtsen_encoding_circuit_n4_l1_local_z

Optional modular replay needs NumPy/SymPy from the public requirements.
It reconstructs the exact parameter point from its stored seed, rebuilds
modular Laurent feature coefficients and tangents, computes the Gram
Jacobian, and checks the saved minor rows/columns, determinant, rank,
support dimensions, conjugate symmetry and every stored activity residue.
It uses the released engine; it is not an independent implementation.
It does not recompute rational determinants or symbolic upper proofs.
The saved fresh replay passed all 60 modular records.

Public replay versus fresh recomputation
----------------------------------------
From the PUBLIC PACKAGE ROOT, use its supported launcher:

  python reproduce.py core-audit

For the pilot, this recomputes formal upper identities and replays saved
feasible interval witnesses at 65 decimal digits. For the extension, the
same command aggregates the stored 30-setting Gram census but DOES NOT
recompute those modular minors or their symbolic upper proofs. It also
performs unit tests/source checks and separate nonlinear checks. Its broad
success flag must not be described as fresh replay of all Gram evidence.

  python reproduce.py core-full --workers 2

This generates fresh pilot results and reruns the full 30-setting Gram
pipeline, including lower bounds and budgeted upper proofs. It also runs
scaling/resource studies beyond this census. The launcher reconstructs
historical path names in a new runs/ directory. Do not invoke legacy
scripts against the renamed public folder without this reconstruction.

Known limits / evidence gaps
---------------------------
1. No standalone 30-setting Gram certificate-only replay command was exposed
   by the public launcher. This supplement adds a targeted lower replay;
   full upper recomputation still uses core-full.
2. The map does not claim newly independent upper-proof verification.
   The stored symbolic provenance and the public full-rerun route are
   identified exactly; current checks reuse the released implementation.
3. The mpmath interval implementation is not proof-assistant verified;
   higher-precision replay is an implementation check, not independent
   mathematical verification.
4. The five rank-zero settings have no positive-rank lower witness; the
   upper-zero identity plus rank nonnegativity is the closing argument.
5. This task did not rerun core-full, rational determinant generation,
   historical learning experiments, or QPU jobs. No rank changes or
   hardware/generalization/speedup claims follow from this index.
