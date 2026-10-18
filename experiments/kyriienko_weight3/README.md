# Kyriienko complete weight-at-most-three readout

Run from the repository root:

```bash
python reproduce.py kyriienko-weight3
```

This freshly computes the missing `(n,k)=(5,3)` numerical entry in the
manuscript's six-pair scan. The command runs seven unit tests, five seeded
experiments and comparisons against all stored arrays. Allow a few minutes.
No network connection or QPU account is used after installing requirements.

## Frozen protocol and result

`protocol.json` was fixed before this calculation. The one-layer circuit is
the pinned sQUlearn Kyriienko chebyshev-tower RY encoding (alpha=2), followed by
local RZ-RX-RZ rotations and the HEA even-then-odd CNOT ladder. There are 15
angles and all 375 nonidentity Pauli observables of weight at most three,
with identity metric. Seeds 3, 42, 123, 777 and 2024 each generate one random
parameter point and 120 random input pairs. No seed was removed or replaced.

All five distance Jacobians have numerical rank 13 at relative tolerances
1e-6 through 1e-10, with an absolute threshold floor of 1e-9.
The thirteenth singular value ranges from 0.2388329319 to 7.2735684042;
the fourteenth is at most 5.5726727211e-14. Full spectra are stored, not
only the ranks. This is numerical evidence, not an exact generic-rank
certificate. In particular it does not prove the odd-n family conjecture,
repair the n=5 block-cover obstruction, or change the 54-setting census.

## Checks and artifacts

- Bit-mask and dense-Pauli implementations agree on all 600 sampled-pair
  Jacobian rows (max absolute error 2.132e-14). Their state-construction
  routines share the same circuit specification; this is not an independent
  mathematical proof.
- The hash-verified, vendored sQUlearn source builds the actual Qiskit circuit.
  On three pairs per seed, Qiskit expectations and derivatives agree with
  the numerical implementation (derivative error at most 1.066e-14).
  `source_host.py` supplies a minimal import host with deferred annotations,
  not a full installed sQUlearn runtime. Vendor source bytes are unchanged.
- Feature-level parameter-shift derivatives are combined with the chain
  rule for squared distance. Applying a two-shift rule directly to squared
  distance would generally be wrong. Centered finite differences with step
  1e-5 agree on those 15 pairs to within 4.965e-9.
- The unmodified archived fast routine reproduces seed 3 and its spectrum.
  It uses the opposite sign for odd-Y feature coordinates. This sign change
  is orthogonal and cancels in the unit-weight squared distance and its
  derivatives. The new implementation uses conventional Pauli Y, checked
  directly against dense matrices and Qiskit. Original bytes are preserved
  under `archived_code/` rather than silently corrected.

`reference_results/arrays.npz` contains theta, inputs, features, distances,
both full Jacobians, singular values, and the Qiskit subset checks for each
seed. `results.json` records errors, source hashes, environment and timing.
It was generated on macOS arm64 with Python 3.9.6, NumPy 1.24.4 and Qiskit
1.2.4. It is not a server or hardware run. The historical server used Python
3.8.10; that statement does not describe this new computation.

The runner's correctness checks do not force rank 13. The public reproduction
entry point separately compares a fresh result to the supplied reference.
Inputs and scientific reference files are never overwritten by a run.
