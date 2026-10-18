# Incremental internal certificate experiment

Frozen executable design: `20261001_boundary_fused_v3`, 1 October 2026. This is a follow-up chosen after earlier coverage results and development pilots, not an initial-study preregistration. The question is whether crossing genuinely later overlapping data gates adds useful savings beyond common compiler, affine, boundary-transport, and rotation-fusion optimizations. This is fixed-model kernel training, not architecture search.

## Development changes before the final design

The first local pilot compared full transport with elementary local tails. Inspection confirmed that three ChebyshevRx symbols disappeared, but one was a boundary case, not a genuine later-data crossing. We therefore added a boundary baseline that permits Clifford/commuting transport, disjoint data, and additive terms from the same native gate, but never crosses a distinct overlapping data gate. A second refinement fuses adjacent same-axis rotations in both strong arms, preventing artificial execution savings from symbolic angle splitting. The two earlier pilot artifacts remain developmental evidence and are not pooled with the final design. These changes strengthen the baseline rather than select a more favorable timing endpoint.

A unit-test fixture initially prepared a family with constant local features, making centered alignment undefined. The implementation correctly rejected it. The zero-active-parameter test was corrected to use a varying-data family; the objective degeneracy guard remains unchanged.

## Scope and data

Primary configurations are frozen-source ChebyshevRx with 2 and 4 qubits, two layers. ChebyshevRx one-layer settings, ChebyshevPQC, and YZ-CX at 2/4 qubits and 1/2 layers are retained as ten controls. No setting is selected according to the new timings. The pinned sQUlearn source host and adapter are copied from library_coverage_v1_2, with vendor bytes and licenses preserved. This is not the entire sQUlearn runtime.

Fifteen input tasks are copied byte-verifiably from the previously audited external server archive, SHA-256 `27751ea6abf0a0ab055b9b964e23eb14fc7cf4a84c56d91e68ce6d6cdd2682f1`. Three datasets are WDBC, Ionosphere, and Auto MPG, each with seeds 42,123,777,999,2024. Each frozen task has 64 training and 64 test rows. Imputation, standardization, two-component PCA, tanh encoding, and target scaling were fit on training data in that earlier protocol. Original metadata and source member paths are included. No preprocessing is refit here. Different seeds are repeated splits of three problems, not 15 independent problems. No validation selection, search, or hyperparameter tuning is performed; zero validation time must not be fabricated as a measured component.

Full training uses ten centered kernel-alignment ascent steps, learning rate 0.1, local XYZ unit-weight RBF gamma 0.5. KRR uses `K + 0.1 I` and a train-target mean intercept, as in the previous real-data protocol; this is an unnormalized ridge convention. All arms use the same full initialization, uniform [-0.3,0.3], seed plus 1009 times configuration index. Inactive coordinates receive zero derivative; initial full vectors are never resized or re-randomized.

## Seven comparison arms

All optimized arms first apply symbolic Qiskit 1.2.4 optimization level 3, approximation_degree 1, seed 71, one process, basis h/cx/rx/ry/rz/rxx/ryy/rzz/rzx. Unsupported operations or nonlinear parameter expressions fail rather than silently fall back. A common level-zero adapter normalization follows compilation.

| Arm | Treatment |
|---|---|
| raw | Original source through split IR; correctness and raw-runtime reference |
| compiler | Optimized source through split IR; representation ablation |
| compiler_fused | Optimized source, recombined angle terms and adjacent same-axis fusion |
| local_tail | Compiler, elementary local-tail deletion, adjacent fusion |
| boundary | Compiler, local tail, restricted boundary/affine transport, adjacent fusion; primary baseline |
| transport | Same boundary pipeline, then unrestricted supported metric transport, adjacent fusion; primary method |
| commutant | Compiler, observable-commutant transport, adjacent fusion; mechanism ablation |

Boundary transport is a subset of the existing sound rule. A parameter occurrence may pass a supported later gate only if the original rule permits it, and may not pass a distinct input-dependent gate overlapping the current transported generator. The same native gate's additive angle terms are exempt from this additional block, retaining the known affine-bias optimization. It does not assert general equivalence completeness.

The elementary local-tail rule is restricted to the exact specified unit-weight full local XYZ readout. It rejects Z-only or anisotropic metrics. Other readouts are checked separately using the metric-aware certifier, not by reusing an invalid local-tail proof. Circuit original indices, reduced occurrences, active symbols, and proof stages are stored separately. A globally shared symbol remains unless every occurrence is removed.

## Timing and backend definitions

Serial per-parameter tangents, batched tangents, and tape adjoints use shared custom NumPy primitives. They are not claimed to be the fastest external library implementations. Full comparison uses five timing repetitions per configuration/dataset/seed/backend. Arm order is block-randomized by fixed seeds; all arms/backends/configurations receive untimed warmup. Preserve all timings, including losses or zero savings.

Cold total includes source construction/verification, compiler, adaptation, elementary simplification, extra certificate, execution-program construction/fusion, input-expression encoding, initialization, ten-step training and ridge prediction. Individual source/compile/adapt/local/boundary/certificate/encoding/train/predict timers are recorded. Program construction and instrumentation are included in total but not all have separate sub-timers. No post-hoc sum of component medians replaces the paired total. Frozen input loading, source numerical validation, warmup, disk writes, condition-number diagnostics, and audit are outside the timer. Preprocessing is outside scope and was already frozen.

Each cold arm actually recompiles independently, so compiler cost is charged to both. The pre-certificate IR is checked for equality across optimized arms during validation. Parameterized programs, not computed kernels or gradients, are reused in the warm experiment. There is no numeric kernel/gradient result cache. Interpreter Pauli lookup caches are common and warmed for all arms.

Primary endpoint is `100*(1 - total_transport/total_boundary)`, paired by configuration/dataset/seed/repetition/backend. Report the two primary configurations and three backends separately. Local-tail comparisons are secondary. Compiler_fused may be faster than a certificate arm and must remain visible; a positive boundary comparison is not universal superiority over all arms. Saved observations support grouped split medians and descriptive ranges. No population significance is inferred from repeated seeds or timing repetitions.

Simulator counters separate gradient calls, derivative entries, state sweeps, tangent lanes, adjoint passes and gate updates. Fusion may leave physical interpreter gate count unchanged despite fewer parameter terms. Nothing here measures QPU shots, physical hardware execution, or peak process memory.

## Measured reuse and workload sizes

Reuse is measured on the two primary configurations, all three datasets, seed 42 only, five repetitions, and all three backends. This is an explicit smaller reuse subset, not the whole 15-task cold study. Both arms prepare once and then execute 20 actual fits with paired changing initializations (`+100000 * reuse_index`). Cumulative wall time is sampled at H=1,2,5,10,20, including the one-time preparation. No break-even point is inferred by cold-timer subtraction. Record the first observed H with a positive reduction; nonmonotonic timings or no observed crossover remain visible.

Full: 2,700 cold paired blocks × 7 arms = 18,900 fits; 90 reuse blocks × 2 arms × 20 fits = 3,600 fits. Total 22,500 fits and 225,000 optimizer steps, plus warmup/validation/audit. Save lossless final Gram matrices with content-addressed deduplication and every training trace. This can produce substantial output; no fixed runtime or disk-size guarantee is made.

Pilot: all 12 configurations, Ionosphere seed42, 64+64 rows, four steps, one timing repetition, H=1,2,5. It has 252 cold fits and 60 reuse fits. Smoke: two primary configurations, eight train/eight test rows sliced separately from the frozen split, two steps, one repetition, H=1,2; 42 cold fits and 24 reuse fits. Pilot/smoke timings are technical checks, not full-study effect estimates.

## Verification and failure policy

Before timing, source Statevector features and source finite differences are compared to IR and all three backend derivatives at nonzero angles for every tested configuration. Validate all optimized pre-certificate IRs are equal. Check Z-only and anisotropic controls without asserting that every parameter in them is active. The common zero-gradient/degenerate-alignment guards are not disabled.

Every timed block checks kernel, full-gradient trace, retained/full theta path, objective and prediction against its original-source-IR reference. Across engines the raw reference paths are also compared. Tolerances: kernel/objective 1e-10, gradient/theta 1e-9, prediction 1e-7; native-source finite differences 2e-7. No model selection is present. Failures preserve partial records, a failure summary and archive; they do not trigger seed substitution or tolerance relaxation.

Post-run audit verifies every manifest hash, saved Gram raw hash, all paired stored paths and timing arithmetic, and expected completion counts. An independent scalar source-kernel derivative path replays repetition zero serial cold references for every configuration/dataset/seed, and repetition zero serial warm fits for both primary configurations. It checks saved increments, not autonomous retraining, and is performed by the same implementation author, not independent human proof review. Other repetitions/backends receive complete paired artifact checks but not this additional scalar replay. Wall-time arithmetic verification is not independent timing reproduction.

The final server acceptance requires both summary.json and audit.json passed=true and expected counts. Python 3.8 server execution is not implied by local Python 3.9 tests. A new full server result is required before manuscript performance claims are updated.
