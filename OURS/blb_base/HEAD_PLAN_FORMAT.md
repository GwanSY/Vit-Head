# Public offline head assignment and runtime plan

`head_assignment.json` is the A2 calibration/audit artifact. It binds the
model SHA-256, four-or-more public samples, per-head FP32/fixed-clear
relative-L2 disturbance, the threshold, measured path cost, final global
logit guard, physical reorder, and `private_dynamic_branch=false`. It is
generated once before inference and is never changed from a private input.

The secure C++ runtime consumes the deliberately smaller materialized
`head_plan.txt` referenced by that JSON. Keeping selection evidence out of the
parser makes the online trusted surface easy to inspect: the runtime only
validates a fixed path/alpha/permutation table and cannot run the selector.

The runtime accepts only `OURS_VIT_HEAD_PLAN_V1`:

```text
OURS_VIT_HEAD_PLAN_V1
layers=12,heads=12,logical_tokens=197,physical_tokens=256
layer,physical_head,original_head,protocol,alpha
0,0,0,softmax_mpc,1
...
```

There must be exactly 144 rows. In every layer, `physical_head` and
`original_head` must each describe the 12-head layout without duplicates.
Protocols are `softmax_mpc`, `relu_norm_mpc`, or `he_linear`. Alpha must be
finite and is legal only for `he_linear`; MPC rows require alpha 1.

The plan is loaded and validated before model/input networking begins. It is
public calibration output and never depends on a private inference input.
Each party writes `head_plan_partyN.csv`, including contiguous group IDs,
group sizes, QK ciphertext counts and 197/256 packing utilization.

A0 does not accept a file: its all-Softmax/original-order plan is built in.
A1-A4 require `--head-plan`. A1 requires `original_head == physical_head`.
A2-A4 require every protocol to occupy at most one contiguous group per layer;
the runtime uses `original_head` to synchronously reorder Q/K/V output blocks,
their biases, and WO input blocks. A1 and A2 execute today. A3/A4 plan
validation remains available with `--validate-plan-only`, while online
execution returns `BLOCKED` instead of silently substituting A2/A0.

## Measured cost CSV

Final assignment calibration requires one row per layer/protocol with this
schema:

```text
layer,protocol,latency_ms,he_ms,mpc_ms,conversion_ms,communication_bytes,rounds,head_count,packing_utilization,multiplicative_depth,ct_ct_multiply,rotations,relinearization,rescale,measured,backend,network_gbps,rtt_ms,execution_domain,cost_stage
```

Only `measured=1`, Phantom GPU, `network_gbps=1`, and `rtt_ms=0.3` are
accepted. A2 linear scaling is explicitly labelled
`execution_domain=mpc_surrogate,cost_stage=A2`; A4 must replace it with a real
CKKS row (`execution_domain=ckks,cost_stage=A4`). HE-linear rows over the
declared depth budget are measured but rejected by selection. Without the
cost file, calibration emits relative-L2 candidates only and cannot claim a
selected assignment.
