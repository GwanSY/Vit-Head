# OURS Native Edge/Cloud Execution

The framework is part of `runtime/ckks_vit_b16_main.cpp` and
`runtime/edge_cloud_runtime.h`. It runs the existing OURS heterogeneous ViT,
not a BLB replacement executable and not a Python backend registered in
lamination. No lamination runtime imports are required.

## Execution Architecture

The lifecycle follows lamination's model: public static plan preparation,
role-specific parameter preparation, cooperative static forward, and release.
The existing CKKS/MPC implementation remains the protocol backend.

```text
Public HeteroPlan + EdgeCloudModelSpec
                 |
         EdgeCloudRuntime (both parties)
                 |
    Cloud / Alice                 Edge / Bob
    local model                   local image
    reorder + weight cache        input sharing
                 |                 |
          existing SCI / CKKS / MPC channels
                 |
    ln1 -> qkv -> heterogeneous norm -> attention output
        -> ln2 -> ffn1 -> gelu -> ffn2 (per block)
                 |
         existing final LN / classifier / reveal
```

The native framework compiles head protocols and alpha values once, checks
the public contract between both roles during setup, enforces static operator
order, and prepares/releases weights only on the cloud side. A0-A4 keep their
existing kernel selection, grouping, selective conversion and fused HE-linear
behavior. DATA/PARAM direction switching is intentionally not ported.

The new setup exchange sends 33 bytes per party (SHA-256 digest plus role).
It is included in setup time/communication, before the online counters begin.
There are no additional network calls in online node dispatch, no Python
callbacks, no tensor serialization and no decrypt/re-encrypt bridge.
Existing streamed per-layer weight caching is retained.

This is an architectural refactor, not a new security proof. The public
contract is a compatibility check, not peer authentication or binary
attestation. It runs after native cryptographic context construction; existing
transport/worker initialization requirements still apply. Both processes must
use the new binary and compatible native runtime settings. It is not wire
compatible with a pre-framework peer.

The original output policy remains public logits to both parties. Existing
public-input benchmark/oracle tools are not a private production service.
For two-host deployment, each host needs its own role's assets and the same
public plan/runtime settings; secure channels still use the existing
`--address`, `--port` and derived worker ports (currently through port + 290).
Do not assume two-host timing equals the loopback/netem benchmark.

## Conversion And Packing

The wire contract also binds `integer_lift_real_pairs_v3`,
`public_compact_bounded_v2`, and `input_ciphertext=seeded_v1`. Both roles must
be rebuilt together; old and new conversion layouts cannot be mixed.
There is one production implementation, not a fast/exact runtime switch.

- CKKS-to-MPC combines prime-field centering and wide-ring sharing into one
  prime-wrap comparison and a full-width random OT correction. It retains
  the 40-bit extension margin and the existing SCI/EMP primitives. Public
  coefficient tests cover both wrap cases, centered endpoints, reversed
  worker roles, and 100/120-bit output rings.
- MPC-to-CKKS extends slot shares once from `l` to `l+40` bits. Alice
  subtracts the sharing modulus locally to choose signed integer
  representatives. Encode keeps those integers in a workspace of
  `l+40+16+log2(N)+2` bits, including the existing 16 fractional guard bits.
  This avoids local FFT wrap and the second coefficient-extension protocol.
  Each signed result is reduced independently into every RNS prime. The
  ordinary statistical lifting boundary event remains; no margin is removed.
  The full signed 37-bit input-range fixture includes endpoints and random
  values, not just small activations.
- QK/V (non-selective boundary), exponent outputs, and ViT FFN1/GELU outputs
  pair adjacent real tensors as `Re(a) + i*Re(b)`. Conjugate projection
  removes any unwanted imaginary input. The projection factor two is
  accounted for in the ciphertext scale. The coefficient domain must retain
  one bit of headroom; changed models still require numerical validation.
  Both roles allocate `ceil(original_ciphertexts/2)` transport ciphertexts.
  Conversion decodes both channels and restores the original tensor order,
  including an odd final ciphertext. Sending only the first half of an
  ordinary real ciphertext vector is NOT this packing operation.
- Softmax compacts public masked key columns before max/exp/normalization.
  In ViT pre-norm execution it also omits padded query rows, whose outputs
  cannot reach a valid token. It restores the original layout with exact
  zeros and retains every valid query and key. Generic callers retain their
  query rows. The row reciprocal uses `s+2` bits after a secure row-level
  denominator floor and reciprocal ceiling; multiplication still returns
  the original output ring. The bounds leave exact Softmax unchanged since
  its shifted denominator is at least one. Generic/ReLU division stays wide.
- ViT pre-norm residuals are added to same-scale MPC shares after conversion.
  No residual is opened. Post-norm paths retain their original HE placement.
- The prior real-coefficient symmetry optimization and repeated LayerNorm
  inverse-sigma import reuse remain enabled. BSGS and direct GPU plaintext
  encoding are retained.
- Fresh client inputs use SEAL symmetric encryption with seeded
  serialization. Bob already owns the secret key; Alice expands the
  transmitted random-polynomial seed using the ordinary SEAL loader.
  This does not transmit the secret key or alter the encryption parameters.
  Evaluated/masked server ciphertexts still send both polynomials.
- Short QKV/FFN affine and rescale launches stay on the caller's CUDA stream.
  An OpenMP join alone does not synchronize workers' per-thread CUDA streams;
  the previous mixed CPU/GPU loops could race the subsequent evaluator or
  host export. The matrix kernels retain their explicitly synchronized
  parallel streams. HE exponent workers synchronize both their inputs and
  outputs before cross-thread consumption.

The historical 498.103927-second run is a timing anchor, not a numerical
acceptance result: its stored validation has secure top-1 705 versus
fixed/clear 248. This observation concerns that ViT implementation/run,
not the correctness of the BLB paper's packing method. Acceptance requires
both measured online performance and numerical agreement. Diagnostic traces
and setup time must not be presented as production online timing. A one-block
forecast is not a measured 12-block result.

The original repository also contains timing-oriented BERT implementation
shortcuts: `BLB/gpu/SCI/tests/bert_large_bolt/ckks_bert.cpp` explicitly labels
its first-half transfer as `simulate x+y*j optimization`, and that directory's
`linear_ckks.cpp` caps rotations and omits feature-reduction rotations. These
are observations about repository code, not a claim that the paper's packing
cannot support ViT. Reusing a timing shortcut is not equivalent to implementing
its stated packed arithmetic on a real checkpoint.

Numerical diagnosis supports one- or two-block A0 prefixes. Trace filenames
include their actual block index, and `compare_a0_trace.py --block-limit 2`
checks the second block and final prefix classifier. Traces open intermediate
values and are never performance-eligible or production-private runs.

## Model Development

1. Change/export weights through the existing `vit_model` and checkpoint path.
   Only the cloud reads weights; only the edge reads the input image.
2. Change head assignment through `HeteroPlan` and the existing offline
   selection/calibration tools. Both roles consume the same public plan.
3. For a new fused block operator, register its name in
   `EdgeCloudModelSpec.block_nodes`, call `execute_node(layer, name, callback)`
   from the model forward, and implement/reuse its native backend operation.
   The callback runs inline with the existing share/ciphertext types.
4. For structural changes, update the model descriptor/revision, model loader,
   head-plan schema and backend shape support together. The current ViT-B/16
   backend still requires 12 layers/12 heads and 197/256 tokens; the framework
   does not magically make its kernels shape-independent.
5. Rebuild BOTH roles, run `ours_vit_edge_cloud_test`, then the existing
   numerical and paired end-to-end regressions. Keep future model forward
   changes inside the static execution interface rather than adding an
   independent runner for each model revision.

## Entry Points

The rebuilt `ours_vit_b16_main` always uses this architecture. Existing
`--party 1/2`, A0-A4 runners and config files remain accepted. Explicit
`--role cloud/edge` is also supported; conflicting role/party flags fail.

Cloud example (in the established native runtime environment):

```bash
ours_vit_b16_main --role cloud --backend gpu --port 31003 \
  --model MODEL --ablation a4 --head-plan PUBLIC_PLAN --output CLOUD_OUTPUT
```

Edge example:

```bash
ours_vit_b16_main --role edge --backend gpu --address CLOUD_HOST --port 31003 \
  --input IMAGE --ablation a4 --head-plan PUBLIC_PLAN --output EDGE_OUTPUT
```

Set each process's `CUDA_VISIBLE_DEVICES` to its separately allocated GPU UUID.
Use only the GPUs allocated by the user. Check idleness before launching;
never use a shared physical GPU for the dual-A100 comparison. Existing paired
container runner accepts `OURS_VIT_PHYSICAL_GPUS=cloud_index,edge_index`.

Normal builds use `scripts/build_vit_b16.sh`, including the native edge/cloud
and numerical test targets. The canonical executable is now
`build/container-gpu/bin/ours_vit_b16_main`. The previous executable and sources
were archived in `outputs/runtime_cleanup_20260916/before_cleanup.tar.gz`.

## Archived Framework Verification

Artifacts: `outputs/edge_cloud_native_20260912/`.

- `before/`: driver snapshot and preserved working executables.
- `paired_build/ours_vit_b16_direct`: pre-refactor OURS driver.
- `paired_build/ours_vit_b16_integrated`: refactored OURS driver.
- Both candidates were compiled in the existing OURS image using the same
  `build/container-gpu` crypto libraries and compiler flags. This is NOT a
  comparison against a substituted BLB model.
- `paired_build/build_manifest.json`: source, header, binary and kernel hashes.
- `cli_verification/verification.json`: linked executable plan/role checks,
  A0/A2/A3/A4, with GPU access disabled. A1 is covered by the C++ runtime test.
- Native C++ test covers A0-A4, cloud-only cache ownership, 12-block ordering,
  extension callbacks, incompatible plans/roles and real two-process socket
  agreement. Its numerical values are mock backend values, not ViT logits.

The temporary driver-only candidate builder has been removed. Rebuild the
canonical executable and its backend together with `scripts/build_vit_b16.sh`.

## Performance Acceptance

The current regression is one complete A0/all-Softmax run on the two GPUs
specified by the user, with the existing model, input, threads, streams and
network settings. Check logits and native operation counts as well as time.
One run does not establish a stable timing median or dataset accuracy.

The historical original-BLB/full-Softmax figure is 498.103927 seconds secure
online, not setup plus online and not complete request wall time. It remains
an absolute performance reference, not the model used in this refactor.
The validation report records the percentage difference from 498.103927
seconds and whether it is within 5%. This is a test criterion, not a measured
result or a guarantee for future model changes.

A full dual-A100 A4 run completed on GPU 2/3 on 2026-09-16. It used
6 Softmax + 6 HE-linear heads per layer, unpacked heterogeneous attention,
and legacy `fast` conversion. Its secure online time was 914.460734 seconds;
secure Top-1 was 235, versus 248 in both references. This is a numerical
failure, not evidence that disabling heterogeneity slows the framework down.
Results are in `outputs/edge_cloud_native_20260916/gpu2_gpu3_full12_01/`.

## Numerical Repair And Baseline Audit

The shared configuration's `group_packing=0` is for A3/A4. `secure_command`
resolves A0 to group packing 1, including
when an assignment file overrides the requested ablation to A0. It does not
change A4 settings or allocator cost probes. A0 uses the built-in all-Softmax
plan, `softmax_he`, and the original attention output operator; it does not
configure heterogeneous layer groups. The native architecture adds only its
setup contract exchange and inline online dispatch.

The 2026-09-16 regression exposed that the historical `baseline_fast` QK
schedule omitted rotations and feature reductions, and AV reused ciphertexts
without extracting their diagonals. These are not equivalent optimizations.
The runtime now has one complete diagonal kernel; the baseline-packing switch
was removed. `blb_fast_attention_packing=0` remains a constant audit field for
existing result readers, not a selectable mode. Full-model numerical and
performance acceptance is still required. AV rotation counters include every
feature ciphertext, which older reports did not count consistently.

The local wide FFT root product uses a split integer product, checked against
an unbounded-integer oracle without lowering precision or security parameters.
Softmax clamps negative CKKS approximation tails before unsigned accumulation
and performs its final shared multiplication in batched MPC, avoiding two
CKKS imports and one export of the attention tensor.

The removed `fast` conversion was a historical compatibility mode, NOT a numerically validated mode.
A public GPU MPC -> CKKS -> MPC test reproduced failure on all 16,384 slots.
That code suppressed the source-to-target scale reduction and
uses the legacy local-share FFT. Merely relabeling A4 as A0 cannot fix this.
The same fixture with the existing `exact` conversion passed, with a maximum
error of 2 raw units (2 / 4096). This primitive result is not a full-ViT or
dataset-accuracy result.

The canonical runtime retains existing cryptographic primitives and uses one
wide/shared-RNS conversion path. Its Softmax maximum batches comparisons at each tree
level: 128 keys require 7 comparison/selection batches, rather than 127
sequential batches. Signed predicate orientation, ring widths, privacy masks,
and the model graph are unchanged. `secure_row_max_test.cpp` tests real SCI
shares, negative values, ties, multiple rows, and odd and padded key counts.
Legacy words no longer emit a false canonical-field PASS log.

Use `scripts/run_vit.py` with `ours_vit_b16/experiments/HAP-ViT.json`.
The temporary candidate builders, numerical suite launcher and five
`edge_cloud*.json` profiles were archived and removed from active code.
There is no conversion-mode selector. For example:

```bash
python3 -B scripts/run_vit.py --gpus 4 6 \
  --output-dir outputs/runtime_cleanup_20260916/a0_gpu4_gpu6
```

The entry point automatically runs the model-level reference check after
secure inference. `native/validation.json` records both parties, operation
counts, final-logit errors and secure online time. A numerical failure returns
nonzero; performance acceptance is recorded separately. The numerical gate
requires matching fixed-reference Top-1, relative L2 <= 0.05, maximum absolute
error <= 0.5, finite logits and matching public outputs from both parties.
Primitive test success alone does not establish full-model correctness.

The historical 498-second binary's source/library identity is different from
current OURS. Its recorded secure prediction was 705, not the reference 248.
Consequently preserving its timing and fixing numerical correctness are two
separate acceptance gates. No claim is made that the repaired full model
currently achieves 498 seconds; this requires measurement and optimization.

The rejected BLB-controller implementation was removed from active OURS code
and archived under `outputs/edge_cloud/rollback_20260912/`. The earlier added
lamination runtime files were also removed; its existing implementation and
historical experiments remain untouched.
