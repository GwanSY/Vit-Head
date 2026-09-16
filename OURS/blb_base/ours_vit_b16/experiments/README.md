# OURS ViT-B/16 experiments

Current entry: `python3 -B scripts/run_vit.py` (from `blb_base/`). It reads
`HAP-ViT.json` by default. See [configuration and stages](RESEARCH2.md).
The sections below document historical cohorts in `scripts/legacy/`.


This directory defines the formal experiment cohorts without changing the
`vit_b16` baseline tree. All formal performance runners require two physical
GPUs and prove active loopback `netem` shaping at 1 Gbps and 0.3 ms RTT.

Important claim boundaries:

- The existing 50-image results are **fixed-clear accuracy**, not secure
  inference accuracy.
- A run without a successful `qdisc_active.txt` audit is **unshaped** and is
  not eligible for the 1 Gbps / 0.3 ms table.
- Intermediate A0 and Packed/Unpacked traces reconstruct validation-only
  tensors and are therefore `timing_eligible=0`.
- `full12_budgets_candidate.csv` and `p2_ablation_candidate.csv` are candidate
  matrices. They become final only after the assignment finalizer binds a PASS
  A0 report and a real secure cohort of at least 50 images.
- The P1 plan generator refuses a Sensitivity-only or Cost-only assignment
  that is byte-for-byte equivalent to Ours, and refuses duplicate Random
  assignments; identical rows are not accepted as independent baselines.

Run from `OURS/blb_base` after rebuilding the immutable `ours` image:

```bash
scripts/legacy/run_ours_a0_container.sh outputs/paper/a0/<run-id>
scripts/legacy/run_ours_three_path_curve.sh <run-tag>
scripts/legacy/run_ours_secure_accuracy.sh a3 <all-softmax-plan> \
  ours_vit_b16/experiments/secure_manifest_50.csv <tag> 50
scripts/legacy/run_ours_packing_ablation.sh a3 <fixed-plan> <tag> 3
scripts/legacy/run_ours_full12_suite.sh \
  ours_vit_b16/experiments/full12_budgets_candidate.csv <tag> 3
scripts/legacy/run_ours_full12_suite.sh \
  ours_vit_b16/experiments/p2_ablation_candidate.csv <tag> 3
scripts/legacy/run_ours_p1_secure_comparison.sh \
  <p1-plan-manifest.json> \
  ours_vit_b16/experiments/secure_manifest_50.csv <tag> 50
```

The secure accuracy and comparison drivers are resumable at sample/run
boundaries. They preserve an incomplete directory instead of deleting or
overwriting it.

The frozen 50-image cohort is `secure_manifest_50.csv` (50 rows, SHA-256
`d4b500ed21cda1d6f5a6a27e708256af45af23d3a02a6bb80ecb9736fb2b87c9`).
