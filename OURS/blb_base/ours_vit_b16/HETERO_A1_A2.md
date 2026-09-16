# A1/A2 implementation boundary

This document is the review map for head-level heterogeneous attention. A0
remains the correctness gate; A1/A2 results must not be reported as final
end-to-end speedups while A0 is FAIL.

## A1: path semantics, original head order

- Fixed assignment parsing/validation: `runtime/hetero_plan.{h,cpp}`.
- Secure path execution: `Bert_ckks::heterogeneous_attention_norm_a1` in
  `runtime/ckks_bert_basic.cpp`.
- Runtime call site: the attention-normalization branch in
  `runtime/ckks_vit_b16_main.cpp`.
- Public reference semantics: `scripts/vit_reference.py`.
- Independent manual-mixed oracle: `scripts/diagnostics/verify_manual_mixed_paths.py`.
- Review-plan generator: `scripts/legacy/make_review_head_plan.py`.

A1 accepts `HeadProtocol` values directly; there are no integer path codes.
It processes each physical head independently and deliberately retains the
full CKKS→MPC→CKKS boundary. `he_linear` therefore denotes the linear-scaling
semantics but is logged as `execution_domain=MPC_surrogate` in A1/A2. This is
not the A4 CKKS implementation.

## A2: offline selection and grouping

- Per-head disturbance calibration and complete-assignment selection:
  `scripts/legacy/calibrate_heterogeneous_heads.py`.
- Formal A3/A4 cost collection:
  `scripts/collect_attention_assignment_costs.py`.
- A2 normalization-only diagnostic collection (not selection-eligible):
  `scripts/legacy/collect_secure_path_costs.py`.
- Reorder oracle: `scripts/verify_head_reorder.py`.
- Runtime grouping: the same secure function as A1 with
  `group_contiguous_protocols=true`; the plan validator requires at most one
  contiguous physical group per path.

The calibrator computes

```text
relative-L2 = ||candidate attention-head context - Softmax context||_2
              / ||Softmax context||_2
```

for FP32 and fixed-clear observations. Formal selection compares only exact
12-Head assignments already measured from QKV/QK through selective conversion,
grouped MPC normalization, AV, W_O, and genuine Algorithm-1 output conversion.
An assignment is eligible only when its exact original-Head protocol vector
passes every disturbance bound. No per-Head division, protocol-group addition,
layer copying, or fixed non-attention cost is used. It evaluates the resulting
fixed plan once with FP32 and fixed-clear logit/accuracy guards.

The selector writes two files:

- `head_assignment.json`: complete immutable audit evidence, model hash,
  disturbance, measured cost, thresholds, physical order, and final guard.
- `head_plan.txt`: the minimal 144-row runtime table referenced by the JSON.

Selection is never linked into `ours_vit_b16_main`; inference only parses the
fixed runtime table before opening secure protocol sockets.

## Per-path secure metrics

Each A1/A2 invocation emits one machine-readable line per executed group:

```text
OURS_HETERO_PATH_PROFILE status=PASS ablation=A1|A2 layer=N
protocol=... physical_head_begin=N head_count=N latency_ms=...
communication_bytes=... rounds=... execution_domain=MPC_CKKS_hybrid
conversion_scope=shared_full_boundary private_dynamic_branch=0
```

For A1/A2 Linear-surrogate groups, `execution_domain=MPC_surrogate`; genuine
A4 rows must say `execution_domain=CKKS`.

For A2 diagnostics, `OURS_VIT_STOP_AFTER_NORMALIZATION=1` executes the real
QKV/CKKS→MPC prefix, the selected grouped normalization kernel, and then exits
synchronously before MPC→CKKS/AV/WO. The cost collector uses only the
`OURS_HETERO_PATH_PROFILE` normalization bucket. Its CSV and assignment JSON
therefore require `cost_scope=a2_normalization_only` and
`conversion_scope=excluded_deferred_to_A3`; zero `conversion_ms` must not be
read as a free conversion. A3 owns grouped conversion accounting.

These rows cannot drive formal selection. The retired `--costs` CLI option now
fails closed; formal calibration requires `--assignment-costs`. Assignment
rows are accepted only when both parties exit zero, the runtime emits the
complete-assignment and real A3/A4 output markers, and metadata proves Phantom
GPU at 1 Gbps / 0.3 ms RTT on two separate dynamically selected physical GPUs.
A shared-GPU run, surrogate marker, copied layer, or missing native CKKS marker
is rejected.

Softmax uses BLB's HE-assisted `exp_he` plus reciprocal engine. ReLU keeps the
secret ReLU/zero-denominator mux in MPC and uses the same HE-assisted
reciprocal; neither path contains the removed full-MPC `div_batch` bottleneck.
A prior wide-ring MPC-to-CKKS experiment and its environment switch have been
deleted from production; only the BLB Algorithm 1 conversion remains.
A2 `he_linear` remains explicitly marked `mpc_surrogate` and is written with
`selection_eligible=0`; A4 must replace it with a genuine CKKS measurement and
full A4 path scope.

The selection report intentionally contains no E2E estimate. The eight-minute
verdict is populated only by a real full 12-layer secure end-to-end run.

## Historical normalization diagnostics (not formal cost rows)

At 1 Gbps / 0.3 ms RTT on two distinct dynamically selected physical GPUs,
the exact measured normalization groups are:

| Group | Latency | Aggregate communication | Rounds |
|---|---:|---:|---:|
| Softmax-12 | 7.308568 s | 274,439,044 bytes | 226 |
| Softmax-11 | 7.123358 s | 251,569,461 bytes | 226 |
| ReLU-1 | 2.899737 s | 13,485,948 bytes | 184 |
| ReLU-12 | 32.594751 s | 164,561,490 bytes | 186 |
| Linear-12 MPC surrogate (ineligible) | 38.414002 s | 89,667,744 bytes | 12 |

These numbers remain useful for normalization implementation review only. They
cannot be summed with a baseline remainder, copied across layers, or used to
select an A3/A4 production assignment.
