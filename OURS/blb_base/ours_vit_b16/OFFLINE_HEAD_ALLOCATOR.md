# Offline security-cost-aware Head allocation

For the configurable research-content-two entry point, use
[`experiments/HAP-ViT.json`](experiments/HAP-ViT.json) and
[`experiments/RESEARCH2.md`](experiments/RESEARCH2.md). Budgets default to
`0.1, 0.5, 1.0` percentage points and can be replaced by any nonempty list of
unique nonnegative values. Each budget exports its own minimum-predicted-cost
eligible plan under `budgets/`; `budget_candidates.csv` identifies duplicate
plans. The older five-policy reports below remain compatibility outputs.

This directory contains the offline allocator and its second-phase validation
driver for the already accepted A3/A4 heterogeneous Attention implementation.
It does not modify selective conversion, native CKKS Linear, or the secure
protocol. It never starts secure inference.

## Data split and real cost observations

`make_composition_probe_suite.py` creates eight fixed one-block plans: four
training compositions and four strict held-out compositions. The default
suite covers S+R, S+L, R+L, and S+R+L in both splits. The plans are measured by
`scripts/run_vit.py --stage costs`, which reuses the existing direct
GPU/HE-MPC runner and `collect_attention_assignment_costs.py`. That command
runs those probes with `stop-after-block=1`, `attention` scope, and also fills
missing complete 12-layer cost anchors.

The existing complete-assignment CSV is validated as direct real benchmark
data. New rows must pass the same provenance checks, use the same secure
binary, and have `cost_unit=direct_assignment_not_per_head`,
`surrogate=0`, `dynamic_selection=0`, separate physical GPUs, and the shaped
1 Gbps/0.3 ms network. A missing probe is represented as
`NOT_MEASURED_NO_REAL_HELDOUT`; it is never filled with a copied, interpolated,
or simulated value.

The cost model used for offline ordering is fitted after the split. For each
target (`latency_ms` and `communication_bytes`) it uses a standardized ridge
regression over measured complete-assignment observations:

```text
f_l(S,R,L) = [
  1, S/12, R/12, L/12,
  active_group_count, SR, SL, RL, SRL,
  (S/12)(R/12), (S/12)(L/12), (R/12)(L/12),
  layer_1, ..., layer_11
]

C_l(S,R,L) = ridge_latency(f_l)
B_l(S,R,L) = ridge_communication(f_l)
C_model(plan) = sum_l C_l(S_l,R_l,L_l)
B_model(plan) = sum_l B_l(S_l,R_l,L_l)
```

The group-size and coexistence terms model shared S/R/L overhead; no fixed
Softmax>ReLU>Linear ratio and no per-Head time sum is assumed. Existing exact
complete rows are used as direct anchors. An unmeasured composition is marked
as a model prediction and is used only for offline search ordering and the
reported predicted cost. It is not a secure-runtime measurement.

`cost_model_holdout.csv` scores only the strict held-out compositions. Its
summary includes latency/communication MAPE, the six strict latency-pair
correct count, and ranking accuracy. The allocator refuses to proceed unless
all four held-out points are real, latency MAPE is at most 15%, and at least
five of the six latency orderings are correct. An incomplete run reports
`NA` for metrics that have no measured held-out pair and emits no Pareto or
secure plan. It still creates the five requested result-table files with their
normal schemas and zero data rows, plus `blocked_result_tables.json`; this
keeps an incomplete experiment distinguishable from a successful empty
frontier without inventing measured values.

`cost_model_validation.csv` is an exact copy of the holdout table for callers
that use the validation name; it is not a second measurement set. The
no-group experiment rejects legacy cost CSVs that lack explicit
`attention_packing=unpacked` and `heterogeneous_head_grouping=0` provenance.
`summarize_composition_probe_status.py` records every attempted container
probe and its real failure reason. `run_clear_sensitivity.py` can recompute
the clear-only 144-entry sensitivity artifacts independently, but it never
creates a cost model or secure plan.

## Head search and clear validation

The clear model starts from all-Softmax. For every layer/head it evaluates a
single ReLU replacement and a single calibrated Linear replacement. Each
Linear `alpha_h` is fitted offline by least squares:

```text
alpha_h = argmin_alpha || alpha * (QK_score_h V_h)
                         - (Softmax_weight_h V_h) ||_2^2
```

For each layer and each composition in the 91-point S/R/L count grid, a
prefix-count DP chooses the concrete original Head IDs with minimum isolated
validation-loss/logit-deviation proxy. Layer options are Pareto-pruned on
proxy disturbance, model-predicted latency, and communication. A second
layer DP builds a small complete-model Pareto frontier; no `3**144`
whole-model enumeration is performed. Every retained candidate and every
matched random baseline is then run through the complete plaintext ViT.

Selection uses the fixed calibration set only. The fixed held-out set is used
only for final accuracy and the Accuracy--Latency Pareto mark. Budgets are
command-line parameters; the default sweep is `0.1, 0.5, 1` pp.
The driver evaluates All-Softmax, Accuracy-only, Cost-only, Balanced, and
Performance-first. Random baselines preserve each layer's exact S/R/L counts
and randomize only original Head IDs with a recorded seed. The final
Performance-first composition has a configurable number (default 20) of held-out random
trials in `random_head_assignment_stats.csv`, including mean/std against Ours.
ReLU is never inserted to make a three-path composition; a zero-ReLU choice is
valid.

`head_sensitivity.csv` contains all 144 isolated Softmax-to-ReLU and
Softmax-to-Linear risks, while `sensitivity_relu_heatmap.csv` and
`sensitivity_linear_heatmap.csv` are each 12x12 layer-by-head matrices.

`run_true_small_oracle.py` exhaustively enumerates 3^H assignments for a
user-selected 4--6-head subset (default four Heads) at one layer. It runs the
complete clear ViT for every assignment on the fixed calibration set, so the
accuracy/risk coordinate is a true full-model value rather than a sum of
isolated-head errors. Its cost coordinate uses a direct measured anchor when
available and otherwise the explicitly labelled composition-model prediction.
The table compares this true exhaustive Pareto set with the prefix-count DP,
records transition/state counts and same-composition risk gaps, and does not
claim a full-model 3^144 brute-force search.

The phase-two command is run first to materialize the five candidates; the
true-oracle command is then run against its fixed calibration manifest and
`isolated_candidate_metrics.csv` to replace `small_oracle_comparison.csv` with
the full-clear comparison and update the experiment report's oracle block.

## Artifacts

The experiment writes:

* `cost_model_holdout.csv`
* `pareto_policies.csv`
* `allocator_baselines.csv`
* `small_oracle_comparison.csv`
* `secure_candidates.csv`

The cost table also records measured rounds, protocol groups, MPC group calls,
native Linear groups, conversion counters, HE time, and HE operator counters.
`unique_secure_plans.csv` is the deduplicated list that may be scheduled for a
later secure check; no 12-layer secure process is launched by this experiment.

It also writes fixed calibration/held-out manifests, the training-observation
audit, isolated clear metrics, five fixed secure-plan JSONs, and a default
`head_assignment.json`. Each assignment includes `layer`, `head`,
`selected_path`, `alpha_h`, `mpc_reference=0`,
`client_decrypt_reencrypt=0`, and `dynamic_selection=false`; the associated
`OURS_VIT_HEAD_PLAN_V1` file is immutable after generation.

The five secure candidates are prepared only, with
`secure_inference_run=false` and `READY_NOT_RUN`. A later secure validation
must use only those fixed plans, deduplicated by plan hash. The predicted costs
must not be presented as final end-to-end secure latency or communication.
