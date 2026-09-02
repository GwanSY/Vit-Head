# OURS ViT-B/16 implementation status

## Donor

- Source: `/home/dm/mind/Vit-Head/BLB`
- Workload: ViT-B/16, 197 logical tokens padded to 256, hidden size 768,
  12 heads, head dimension 64, 12 transformer blocks.
- Backend: Phantom GPU CKKS plus SCI MPC/VOLE.
- Donor runtime source hashes were checked against this fork before the first
  OURS edit; every `vit_b16/runtime/*.cpp` and `*.h` matched byte-for-byte.

Generated build trees, Python caches, SCI install directories and historical
outputs were intentionally not copied. The checkpoint and input are reused
through read-only relative symbolic links.

## Status

| Stage | Status | Evidence |
|---|---|---|
| Clean BLB source fork | PASS | runtime/source directory comparison |
| Phantom GPU CKKS build | PASS | `build/gpu/bin/ckks_vit_b16_main` |
| A0 correctness gate | FAIL / PAUSED | secure full-Softmax numerical mismatch remains; no full experiment is authorized while paused |
| A0/OURS target separation | PASS | both executables link the same local Phantom/SCI backend |
| Public offline head-plan | PASS | 12×12 shape/permutation/protocol/alpha validation and CSV audit |
| FP32/fixed calibration graph | PASS (implementation) | three paths; A0 explicit-plan block-0 oracle is bit-exact |
| A2 calibration candidates | PASS | four public inputs, 144 heads, FP32 and fixed-clear relative-L2 |
| A2 formal eligible-cost evidence | PASS | exact Softmax-12, Softmax-11, and ReLU-1 groups on separate GPUs at 1 Gbps / 0.3 ms; Linear remains ineligible until A4 CKKS |
| A2 fixed assignment | PASS / NO-SWITCH | V2 JSON selects 144 Softmax heads because the only admissible ReLU split is slower as a complete two-group plan |
| A1 mixed-path structure | PASS (block 0) | 4 Softmax + 4 ReLU + 4 Linear-surrogate heads, 12 public invocations |
| A1 secure/fixed numerical oracle | FAIL | secure top-1 585 vs heterogeneous fixed-clear top-1 447 |
| Head reorder oracle | PASS (12 blocks) | fixed max_abs=0; FP32 max_abs=8.18e-14 through WO/logits |
| A2 selector implementation | PASS (repaired) | measured group-size costs, BLB-anchored E2E estimate, two-GPU gate, Linear MPC surrogate excluded |
| A3 selective conversion | PASS (structural, 12-block run) | only assigned MPC groups cross CKKS↔MPC; Linear pairs remain CKKS |
| A4 native CKKS Linear | PASS (structural, 12-block run) | real QK→AV→WO, alpha fused into WO, no client re-encryption/MPC reference |
| A3-A4 numerical acceptance | FAIL / BLOCKED BY A0 | the paused A0 secure graph is already outside the fixed-clear LayerNorm domain |
| A3-A4 performance objective | FAIL (current assignment) | A4 10 Softmax + 2 Linear is 524.750 s vs 537.982 s A3, but both exceed 8 minutes |
| Full 12-block A3-A4 | PASS (diagnostic structure) / performance FAIL | both real current-image runs completed; accuracy gate remains closed by A0 |

The immutable `vit_b16/runtime` remains the donor A0. Fork-specific work is in
`ours_vit_b16`: the offline plan/parser, FP32/fixed calibration, synchronized
head reorder, grouped MPC normalization, selective A3 conversion, and native
A4 CKKS Linear path. `ours_vit_b16_main` runs A0--A4. A3/A4 reject invalid
plans and do not fall back to an MPC Linear surrogate or client bridge.

The historical A1 manual-mixed diagnostic is stored under
`outputs/a1_a2_review/a1_manual_mixed_secure_retry_gpu2_gpu3`. Both parties
returned zero. It executed four heads on each path with no private dynamic
branch. The unshaped attention-prefix latency was 396.087543 s and party-1
communication was 1,047,453,507 bytes. These are correctness/structure
metrics, not formal performance results. The independent public path oracle
passed exactly, while the secure final-logit comparison failed (max-abs
1.715576171875, RMSE 0.5562445472600592 versus fixed-clear).

The four-sample disturbance candidates remain valid and are under
`outputs/a1_a2_review/calibration_candidates_fixed_exact`. ReLU maximum-of-
FP32/fixed relative-L2 admits 1/144 heads at threshold 0.25, 13/144 at 0.30,
and 84/144 at 0.50. Linear admits no head at 0.50. The historical assignment
at `outputs/a1_a2_review/a2_selection_rL2_0p25/head_assignment.json` selected
143 Softmax heads and layer-10 original head 3 as ReLU, but is now invalidated:
its costs came from the wrong slow engine, a shared physical GPU, and a
12-head-cost/12 extrapolation. It must not be used for performance or final
inference. Its disturbance/global-guard numbers remain diagnostic only.

The 349.999 s Softmax, 159.452 s ReLU, and 42.256 s Linear-surrogate rows are
invalid historical measurements. The first two used the removed full-MPC
division bottleneck; all three shared one physical GPU between both parties.
The 69.735/70.000 minute extrapolation derived from those rows is deleted from
the active model.

The repaired production Softmax again uses BLB's HE-assisted `exp_he` and
reciprocal path. ReLU retains MPC ReLU/zero handling but uses the same
HE-assisted reciprocal instead of full-MPC `div_batch`. The experimental
wide-ring conversion and its runtime environment switch were physically
removed; production has only BLB Algorithm 1 conversion. The collector accepts
only 1 Gbps / 0.3 ms runs on two separate dynamically selected GPUs and stores
`C(protocol, group_head_count)` directly. Linear MPC-surrogate rows have
`selection_eligible=0`; only an A4 CKKS row may select Linear.

Fresh formal single-layer evidence from the final `ours` image is under
`outputs/a1_a2_review/a2_finalimg_softmax12_formal_20260831` and
`outputs/a1_a2_review/a2_finalimg_softmax11_relu1_formal_20260831`,
with homogeneous ReLU and Linear-surrogate audits in the adjacent
`a2_repaired_blb_fast_relu12_formal_20260831` and
`a2_repaired_blb_fast_linear12_surrogate_formal_20260831` directories.
The immutable image/binary binding is recorded in
`outputs/a1_a2_review/a2_final_image.txt`.
Softmax-12 measured 7.308568 s, 274,439,044 aggregate bytes, and 226 rounds.
The split plan measured 7.123358 s for Softmax-11 plus 2.899737 s for ReLU-1,
so its 10.023095 s normalization is slower than one Softmax group.
ReLU-12 measured 32.594751 s; Linear-12 MPC surrogate measured 38.414002 s
and remains selection-ineligible.

The replacement immutable assignment is
`outputs/a1_a2_review/a2_final_assignment_rL2_0p25/head_assignment.json`.
It selects 144 Softmax heads and has zero FP32/fixed-clear logit disturbance on
the four public samples. All estimates are anchored to the measured BLB run:
498.104 s E2E, 97.486694 s normalization, and 400.617306 s
non-normalization. The repaired estimate is 488.320127 s (8.138669 min), a
1.964% improvement, but `under_eight_minutes=false`. The current BLB-fast
12-block diagnostics below show real A3/A4 savings, but additional savings are
still required for the strict target. The full runs are performance/structure
diagnostics only while A0 remains paused; they do not authorize an accuracy
claim.

The A2 all-Softmax reverse-order regression is stored under
`outputs/a2_gpu2_reverse_all_softmax_block1_20260830`. Its exact-plan oracle
passed before sockets were opened (fixed max_abs=0, FP32 max_abs=1.43e-14).
The secure run used one grouped Softmax invocation and retained A0's block
communication (361426082 bytes/party), C2M/M2C counts (402/672), ct-ct
multiplications (6144), and rotations (1968). Its 46.739161 s unshaped latency
is diagnostic only.

Build command: `./scripts/build_vit_b16.sh`. Plan parser test:
`./build/gpu/bin/ours_vit_hetero_plan_test`.

## A3/A4 real-path evidence (2026-09-01)

The active A3/A4 implementation is in
`ours_vit_b16/runtime/ckks_bert.cpp` and
`ours_vit_b16/runtime/linear_ckks.cpp`. A4 gives he1 two additional modulus
levels for native QK→AV→WO. The fixed public `alpha_h` is fused into the
corresponding W_O rows with 20 extra fractional bits. Softmax/ReLU groups alone
cross CKKS↔MPC; Linear Heads retain the real he1 QK/V ciphertexts. Logged
invariants explicitly require `client_decrypt_reencrypt=0`, `mpc_reference=0`,
and `surrogate=0`.

There was a concurrent writer to this same workspace during the review. The
shared-he1/single-output-conversion candidate was compiled and measured under
immutable image `sha256:70593a5...`, but another Codex session subsequently
replaced the active source, binary, and `ours:latest` tag with its source-
synced split-he2/he1, two-real-output-conversion variant. That current variant
still implements genuine A3 selective conversion and genuine A4 CKKS Linear;
it does not contain the single-output optimization. The runner now resolves
the mutable tag once and executes the immutable image ID, so recorded runs are
not silently changed by a concurrent rebuild. Do not infer that the shared-he1
candidate is the current worktree implementation from its result row below.

All formal measurements below used two distinct dynamically selected A100
GPUs (physical 0/1 for these runs) and container-local netem
`delay 149us rate 1Gbit`, i.e. about 0.3 ms RTT and 1 Gbps. They are complete
one-block online executions, not per-Head cost extrapolations:

| Fixed plan | Online block | Party-1 online bytes | Rounds | C2M | M2C | Conversion ciphertexts |
|---|---:|---:|---:|---:|---:|---:|
| A3 All-Softmax (12) | 61.683 s | 385,535,708 | 663 | 444 | 684 | 1,128 |
| A4 Softmax 11 + Linear 1, shared he1 | 71.242 s | 372,857,223 | 659 | 424 | 636 | 1,060 |
| A4 Softmax 4 + ReLU 4 + Linear 4 | 81.458 s | 337,137,416 | 859 | 328 | 504 | 832 |
| A4 Softmax 8 + Linear 4, pair-aligned | 61.852 s | 337,950,712 | 659 | 344 | 508 | 852 |

The pair-aligned 8+4 run is the clean A4 mechanism comparison. Relative to
the same current-image All-Softmax block it cuts online communication 12.34%,
C2M 22.52%, and conversion ciphertexts 24.47%, but latency is 0.274% higher.
Relative to the earlier real two-output-conversion implementation of the same
8+4 plan (76.501 s), shared he1 plus one output conversion is 19.15% faster.
Thus the real A4 cost row must include the extra he1 modulus-level cost: the
current implementation does not yet justify selecting Linear for latency.
The 4/4/4 result also shows that a separate ReLU MPC group is too expensive
for this assignment despite lower conversion volume.

BLB packs two Heads in the two ciphertext halves. The A4 review-plan generator
therefore now accepts only an even `--linear-heads-per-layer` count (default 2)
and places selected Linear Heads in one trailing contiguous group. A single
Linear Head is useful for wiring validation but cannot remove a complete QK/V
conversion pair.

Numerical A3/A4 acceptance is intentionally not marked PASS. The independent
A4 attention trace has matching Alice/Bob openings, but comparison to
fixed-clear fails because the paused A0 path is already numerically invalid
before this boundary; its opened block input is outside the fixed-clear
LayerNorm lookup domain. The four-sample calibration also has no Linear Head
under the current 0.25 relative-L2 threshold. Consequently no 12-block secure
run or accuracy claim is made. Full-network execution remains gated on A0 and
single-block numerical acceptance, as required by the staged protocol.

Detailed review notes and exact result paths are in
`ours_vit_b16/HETERO_A3_A4.md`.
