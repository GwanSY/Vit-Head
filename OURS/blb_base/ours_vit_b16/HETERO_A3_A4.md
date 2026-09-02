# A3/A4 implementation and acceptance record

## Scope and hard constraints

- A3/A4 use the BLB Phantom GPU CKKS and SCI MPC implementations.
- No client decrypt/re-encrypt bridge is present.
- `he_linear` never calls the MPC Linear normalization surrogate.
- Head assignment is a fixed public file validated before sockets open.
- Formal runs use two distinct dynamically selected GPUs and container-local
  1 Gbps / approximately 0.3 ms RTT netem.
- Full 12-block performance diagnostics may run while A0 is paused, but their
  predictions are not accuracy evidence.

The active source-synchronized `ours:latest` image is
`sha256:6368d4c1f510d87a9eb6b43fe5cad62f2e67f00af1c271aba94bdac8e6af607a`.
Its container-built main binary SHA-256 is
`ecc7af46323e17b1319b0aab98092a48835f8b3235995ea7f109d8636f18bd5d`.
It uses split BLB-he2/native-he1 output contexts and two genuine output
conversions for a mixed layer. It also fail-closes the direct-cost collector
unless the measured BLB-fast QK/AV markers are present. The current A3
single-layer measurement used image
`sha256:67d15dfb1dc44d31bf0ed0019021312c44077e4e89d3ba2a41fdcc8f656270b7`
with binary
`9e4b75bc6c7396f448b0760e566c2ebcda15af76c3c5053702a857d56b97f67f`;
the subsequent binary change only adds the explicit
`blb_fast_attention_packing=1` run-metadata field, while the image also
contains the stricter collector. Neither changes the measured secure graph.

## Review map

- `runtime/hetero_plan.{h,cpp}`: fixed-plan validation and A3/A4 protocol
  restrictions.
- `runtime/ckks_bert.cpp`: Head masks, selective grouped conversion, A3 AV/WO,
  A4 native QK/AV/WO, real output conversions, and counters.
- `runtime/linear_ckks.cpp`: split W_O preprocessing, signed 37-bit checkpoint
  handling, `alpha_h` fusion, and A4 he1 modulus schedule.
- `scripts/make_a4_sparse_review_plan.py`: calibration-based review plans with
  complete two-Head packing pairs. This is not the final A2 selector.
- `scripts/run_ours_a3_a4_container.sh`: head-reorder oracle, dynamic GPU
  selection, formal netem, and immutable Docker image-ID execution.
- `scripts/run_ours_vit_b16.sh`: secure two-party runner and full derived MPC
  port-range preflight using portable POSIX awk.
- `scripts/collect_attention_assignment_costs.py`: fail-closed collection of
  complete, directly measured assignment rows; no per-Head division or layer
  copying. It also rejects the removed full-diagonal experimental path.
- `scripts/summarize_a3_a4_run.py`: A3/A4-specific full-network validation and
  reporting. It does not label OURS runs as the original BLB protocol.

## A3 selective conversion

For each contiguous Softmax/ReLU group, only active packed Head halves are
exported to MPC. After normalization, only those group shares are imported
back to CKKS. A packed pair is active when either of its two Heads needs MPC;
inactive halves are public zeros with matching ciphertext metadata. Linear
Heads are absent from both selective boundaries.

The active pair-aligned 10 Softmax + 2 Linear run records:

```text
OURS_A3_SELECTIVE_CKKS_TO_MPC ... head_count=10 qk_ciphertexts=40 v_ciphertexts=10
OURS_A3_SELECTIVE_MPC_TO_CKKS ... head_count=10 qk_ciphertexts=40 v_ciphertexts=10
OURS_A3_HE_PATH_PROFILE ... active_heads=10 execution_domain=he2_grouped
```

This is real BLB Algorithm-1 conversion. No conversion count is inferred from
per-Head measurements.

## BLB-fast QK/AV compatibility isolation

A3/A4 now invoke the exact fast diagonal-reuse schedule used by the measured
498.104-second BLB baseline. The previous full-diagonal experiment remains
available only to the paused A0 diagnostic path; it is not reachable from A3
or A4 and its rows are rejected by the direct-cost collector.

The fast AV path consumes the selector's public modulus level once on the 60
compact QK/V source ciphertexts before aliases are created. This is a CKKS
modulus switch only: it adds no multiply, rescale, conversion, decryption, or
MPC call. The unchanged W_O cache therefore still receives chain 3 / scale
2^69 on the MPC-normalized branch. Native A4 consumes the analogous public
level inside he1 and remains CKKS throughout QK -> AV -> W_O.

```text
OURS_BLB_QK_PACKING ... mode=baseline_fast rotation_cap=122 mpc_surrogate=0
OURS_BLB_FAST_AV_LEVEL ... mod_switch_only=60 extra_rescales=0 extra_multiplies=0
OURS_BLB_AV_PACKING ... mode=baseline_fast extra_mask_multiplies=0 extra_rescales=0
```

BLB-fast packing is not numerical acceptance: its markers explicitly record
`a0_numerical_validation=PENDING`, because A0 remains paused.

## A4 native CKKS Linear

Native Linear Heads retain the real he1 QK and V ciphertexts. V is aligned to
QK using one public value-one multiplication and the single added A4 level.
The native QK scale is solved from the actual final W_O rescale prime so that
`QK -> AV -> W_O` reaches the one-prime, scale-2^49 Algorithm-1 boundary with
exactly one added level. `alpha_h` is fused into Linear W_O rows offline.

MPC-normalized Heads remain on the unchanged BLB he2 chain. In a mixed layer,
he2 and he1 contributions are exported independently through two genuine
Algorithm-1 conversions and added as MPC shares. There is no cross-context
decrypt/re-encrypt bridge and no MPC Linear reference:

```text
OURS_A4_NATIVE_CKKS ... alpha_fused_into=WO client_decrypt_reencrypt=0 mpc_reference=0
OURS_A4_HE1_CHAIN ... data_level_extension=1 modulus_entries=9
OURS_SELECTIVE_OUTPUT_CONVERSION ... stage=a4_mpc_heads_wo ciphertexts=12
OURS_SELECTIVE_OUTPUT_CONVERSION ... stage=a4_linear_heads_wo ciphertexts=12
OURS_A3_A4_ATTENTION_OUTPUT ... a4_cross_context_merge=MPC_additive_after_two_real_conversions surrogate=0
```

## Historical formal one-block results

All rows are directly measured complete Transformer blocks, but they predate
the active BLB-rotation restoration and one-level A4 chain. They must not be
used as current cost rows.

| Plan | Seconds | Online bytes | Rounds | Total C2M | Total M2C | Total conversion cts |
|---|---:|---:|---:|---:|---:|---:|
| A3: 12 Softmax | 61.683193 | 385,535,708 | 663 | 444 | 684 | 1,128 |
| A4: 11 Softmax + 1 Linear | 71.241806 | 372,857,223 | 659 | 424 | 636 | 1,060 |
| A4: 4 Softmax + 4 ReLU + 4 Linear | 81.457930 | 337,137,416 | 859 | 328 | 504 | 832 |
| A4: 8 Softmax + 4 Linear, pair-aligned | 61.852177 | 337,950,712 | 659 | 344 | 508 | 852 |

Evidence directories:

- `outputs/a3_a4_review/a3_all_softmax12_block1_formal_current_20260831`
- `outputs/a3_a4_review/a4_shared_he1_sparse11_softmax1_linear_block1_formal_20260831`
- `outputs/a3_a4_review/a4_manual_grouped_4softmax4relu4linear_block1_formal_20260831`
- `outputs/a3_a4_review/a4_pair_aligned_8softmax4linear_shared_he1_locked_block1_formal_20260901`

The final pair-aligned 8+4 result used immutable image ID
`sha256:70593a5f5b5da08204430f10786c5ea190bfe0fbb77a97eb452a663bdc6f8533`
and binary SHA-256
`319c35969ac36f020c4f3f9a649f689d3ab2ae49651d4436858252427332af10`.
The runner resolves the mutable `ours` tag once and passes that immutable ID
to `docker run`, preventing concurrent image rebuilds from changing a run.

Against the same current-image All-Softmax block, pair-aligned 8+4 reduces
online communication by 12.34%, C2M by 22.52%, and conversion ciphertexts by
24.47%; its latency is 0.274% higher. Against the earlier real two-output-
conversion 8+4 implementation (76.501461 s), it is 19.15% faster. These are
assignment-level measurements, not an additive cost projection.

## Current BLB-fast single-layer result

The current All-Softmax A3 result is a directly measured complete secure
attention assignment on physical GPUs 0/1 and the formal 1 Gbps / 0.3 ms RTT
network. Both parties exited zero.

| Metric | Current A3 |
|---|---:|
| Complete assignment | 26.451084 s |
| Complete secure attention block | 28.025664 s |
| QKV HE | 2.052407 s |
| QK HE | 1.927580 s |
| AV HE | 2.896494 s |
| W_O HE | 0.681610 s |
| Assignment communication, Alice + Bob | 639,712,007 B |
| C2M / M2C | 312 / 528 |
| Assignment rotations / rescales | 4,320 / 1,752 |

Evidence:

- `outputs/a3_a4_review/a3_all_softmax_attention1_blb_fast_level_v2_20260901`
- `outputs/a3_a4_review/a3_all_softmax_attention1_blb_fast_level_v2_20260901/direct_assignment_cost_strict.csv`

This removes the earlier QK/AV regression (approximately 130/117 seconds per
layer in the full-diagonal branch) before any full-network rerun.

## Current BLB-fast full-network result

The current image was run end to end for all 12 transformer blocks with the
same fixed 1 Gbps / 0.3 ms RTT netem and dynamically selected physical GPUs.
Both parties exited zero and the structural summaries passed. These runs are
performance/structure diagnostics only; A0 numerical acceptance remains
paused.

| Metric | A3 all Softmax | A4 10 Softmax + 2 Linear/layer | A4 change |
|---|---:|---:|---:|
| Secure end-to-end | 537.982044 s | 524.749614 s | -2.460% |
| Setup + secure, slower party | 800.943315 s | 794.003233 s | -0.867% |
| Transformer blocks wall | 531.797750 s | 517.790360 s | -2.634% |
| Online communication, Alice + Bob | 15,804,193,756 B | 15,017,766,138 B | -4.976% |
| Total communication, Alice + Bob | 19,418,110,031 B | 18,870,964,236 B | -2.818% |
| C2M / M2C | 5,236 / 8,136 | 4,780 / 7,152 | -456 / -984 |
| Conversion ciphertexts | 13,372 | 11,932 | -1,440 |
| Head executions, Softmax / ReLU / Linear | 144 / 0 / 0 | 120 / 0 / 24 | — |
| Secure prediction (not an accuracy claim) | 14 | 306 | — |

Evidence:

- `outputs/a3_a4_review/a3_all_softmax_full12_blb_fast_current_20260901`
- `outputs/a3_a4_review/a4_pair_aligned_10softmax2linear_full12_blb_fast_current_20260901`

The secure predictions differ between A3 and A4, and neither is promoted to an
accuracy result because the A0 numerical gate is still closed.

## Removed full-diagonal matched full-network results

Both rows use the same immutable image, GPUs 0/1 selected immediately before
each run, and 1 Gbps / 0.3 ms RTT. Times are measured secure-graph wall time;
communication is Alice plus Bob. A4 uses one pair of Linear Heads in every
layer (120 Softmax + 24 Linear Head executions over the network).

| Metric | A3 All-Softmax | A4 10 Softmax + 2 Linear/layer | A4 change |
|---|---:|---:|---:|
| Secure E2E | 734.706231 s | 726.316680 s | -1.142% |
| Assignment sum | 511.136099 s | 506.507793 s | -0.905% |
| Communication | 19.418116 GB | 18.870954 GB | -2.818% |
| HE time | 352.784752 s | 369.021237 s | +4.602% |
| Conversion time | 34.974670 s | 32.965640 s | -5.744% |
| MPC nonlinear bucket | 165.691649 s | 148.829766 s | -10.177% |
| CKKS-to-MPC | 5,236 | 4,780 | -8.709% |
| MPC-to-CKKS | 8,136 | 7,152 | -12.094% |
| Conversion ciphertexts | 13,372 | 11,932 | -10.769% |
| Rotations | 566,572 | 568,468 | +0.335% |
| Rescales | 72,463 | 71,599 | -1.192% |

Evidence:

- `outputs/a3_a4_review/a3_all_softmax_full12_blb_rotation_active_20260901`
- `outputs/a3_a4_review/a4_all_layers_pair_full12_one_level_blb_rotation_active_20260901`
- `outputs/a3_a4_review/direct_assignment_costs_a3_a4_full12_active_20260901.csv`

These measurements are retained only as regression evidence. Their image
used the removed full-diagonal QK/AV experiment, so the strict collector now
rejects them as current A2 cost rows.

The directly measured A2 rerun at relative-L2 threshold 0.25 selected 144
Softmax Heads. No Linear Head is disturbance-eligible (best max-over-oracles
relative-L2 is 0.523137). The result therefore preserves the reference path
instead of selecting the faster measured A4 assignment at unacceptable
disturbance. Its V3 report explicitly sets all extrapolation flags to false
and leaves the eight-minute verdict to the full secure run.

### Current verdicts

- A3 selective conversion on BLB-fast packing: **PASS** for the current real
  single-layer run; one real output conversion, no surrogate.
- A4 native CKKS Linear on BLB-fast packing: **PASS** for the current-image
  12-block structural run; its secure-only time is 524.749614 s.
- Full 12-layer under-eight-minute target: **FAIL** for the current assignment;
  A4 is 524.749614 s versus A3 at 537.982044 s.
- A3/A4 numerical accuracy: **FAIL / blocked by paused A0**. Party prediction
  agreement and clear calibration are not reported as secure accuracy.
