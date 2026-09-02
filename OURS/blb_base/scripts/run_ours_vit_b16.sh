#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 cpu|gpu OUTPUT_DIR PORT [ALICE_GPU_ID [BOB_GPU_ID]]" >&2
  exit 2
fi

backend=$1
output_dir=$2
port=$3
alice_gpu_id=${4:-}
bob_gpu_id=${5:-$alice_gpu_id}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
binary="$root/build/gpu/bin/ours_vit_b16_main"
model=${BLB_VIT_MODEL:-$root/checkpoints/vit_b16/bolt_model.txt}
input=${BLB_VIT_INPUT:-$root/inputs/vit_b16/bob_image.txt}
stop_after_block=${BLB_VIT_STOP_AFTER_BLOCK:-12}
timeout_s=${BLB_VIT_TIMEOUT_SECONDS:-21600}
omp_threads=${BLB_VIT_OMP_THREADS:-64}
gpu_direct_plaintext=${BLB_VIT_GPU_DIRECT_PLAINTEXT:-1}
gpu_linear_streams=${BLB_VIT_GPU_LINEAR_STREAMS:-8}
network_mode=${OURS_VIT_NETWORK_MODE:-netem}
ablation=${OURS_VIT_ABLATION:-a0}
head_plan=${OURS_VIT_HEAD_PLAN:-}
prevalidated_reorder_binding=${OURS_VIT_PREVALIDATED_REORDER_BINDING:-}

if [[ "$output_dir" != /* ]]; then output_dir="$root/$output_dir"; fi
if [[ "$backend" != cpu && "$backend" != gpu ]]; then
  echo "backend must be cpu or gpu" >&2
  exit 2
fi
if [[ "$backend" == gpu && -z "$alice_gpu_id" ]]; then
  echo "GPU IDs are required for the gpu backend" >&2
  exit 2
fi
if [[ ! "$port" =~ ^[1-9][0-9]*$ || "$port" -gt 32767 ]]; then
  echo "PORT must be an integer in [1, 32767]" >&2
  exit 2
fi
# NonLinear owns 64 workers at base+1+3*i and each worker opens auxiliary
# channels at +50/+100.  Validate the entire derived range rather than only
# the main socket; a collision otherwise appears several minutes into setup.
max_worker_port=$((port + 290))
if [[ "$max_worker_port" -gt 32767 ]]; then
  echo "PORT leaves insufficient room for the 64 MPC worker channels" >&2
  exit 2
fi
if awk -v low="$port" -v high="$max_worker_port" '
    function hex_to_dec(value, result, digit, i) {
      result = 0
      value = toupper(value)
      for (i = 1; i <= length(value); ++i) {
        digit = index("0123456789ABCDEF", substr(value, i, 1)) - 1
        if (digit < 0) return -1
        result = result * 16 + digit
      }
      return result
    }
    NR > 1 {
      split($2, local_address, ":")
      split($3, remote_address, ":")
      local_port = hex_to_dec(local_address[2])
      remote_port = hex_to_dec(remote_address[2])
      if ((local_port >= low && local_port <= high) ||
          (remote_port >= low && remote_port <= high)) found = 1
    }
    END { exit found ? 0 : 1 }
  ' /proc/net/tcp /proc/net/tcp6; then
  echo "refusing occupied MPC port range: $port-$max_worker_port" >&2
  exit 3
fi
if [[ ! "$omp_threads" =~ ^[1-9][0-9]*$ || "$omp_threads" -gt 64 ]]; then
  echo "BLB_VIT_OMP_THREADS must be an integer in [1, 64]" >&2
  exit 2
fi
if [[ "$gpu_direct_plaintext" != 0 && "$gpu_direct_plaintext" != 1 ]]; then
  echo "BLB_VIT_GPU_DIRECT_PLAINTEXT must be 0 or 1" >&2
  exit 2
fi
if [[ ! "$gpu_linear_streams" =~ ^[1-9][0-9]*$ ||
      "$gpu_linear_streams" -gt 16 ]]; then
  echo "BLB_VIT_GPU_LINEAR_STREAMS must be an integer in [1, 16]" >&2
  exit 2
fi
if [[ "$network_mode" != netem && "$network_mode" != none ]]; then
  echo "OURS_VIT_NETWORK_MODE must be netem or none" >&2
  exit 2
fi
if [[ ! "$ablation" =~ ^[aA][0-4]$ ]]; then
  echo "OURS_VIT_ABLATION must be a0, a1, a2, a3, or a4" >&2
  exit 2
fi
if [[ "${ablation,,}" != a0 && -z "$head_plan" ]]; then
  echo "OURS_VIT_HEAD_PLAN is required for A1-A4" >&2
  exit 2
fi
if [[ -n "$head_plan" && ! -f "$head_plan" ]]; then
  echo "missing head plan: $head_plan" >&2
  exit 2
fi
for path in "$binary" "$model" "$input"; do
  if [[ ! -e "$path" ]]; then echo "missing required path: $path" >&2; exit 2; fi
done

mkdir -p "$output_dir"
# The secure parties run as root inside the reference container, while the
# float/fixed-clear verifier is intentionally run from the mounted host tree.
# Keep the selected result directory writable by that verifier.
chmod a+rwx "$output_dir"
if [[ "$backend" == gpu ]]; then
  {
    echo "format=OURS_VIT_GPU_SELECTION_V1"
    echo "selection_time=$(date --iso-8601=ns)"
    echo "physical_gpus=$alice_gpu_id,$bob_gpu_id"
    if [[ "$alice_gpu_id" == "$bob_gpu_id" ]]; then
      echo "party_topology=shared_physical_gpu"
    else
      echo "party_topology=separate_physical_gpus"
    fi
    echo "selection_source=runtime_arguments_after_nvidia_smi_check"
  } > "$output_dir/gpu_selection.txt"
fi

# A2 and later physically reorder checkpoint head blocks. Refuse to open the
# secure sockets unless the exact model/input/plan permutation first passes
# the FP32 and fixed-clear WQ/WK/WV/QK/norm/AV/WO/logit oracle.
if [[ "${ablation,,}" =~ ^a[2-4]$ ]]; then
  if [[ -n "$prevalidated_reorder_binding" ]]; then
    if [[ ! -f "$prevalidated_reorder_binding" ]]; then
      echo "missing prevalidated reorder binding: $prevalidated_reorder_binding" >&2
      exit 2
    fi
    binding_value() {
      local key=$1
      local value
      value=$(awk -F= -v key="$key" '$1 == key {print substr($0, index($0, "=") + 1)}' \
        "$prevalidated_reorder_binding")
      if [[ -z "$value" ]]; then
        echo "prevalidated reorder binding lacks $key" >&2
        exit 2
      fi
      printf '%s' "$value"
    }
    require_binding_match() {
      local key=$1
      local expected=$2
      local actual
      actual=$(binding_value "$key")
      if [[ "$actual" != "$expected" ]]; then
        echo "prevalidated reorder binding mismatch for $key" >&2
        exit 2
      fi
    }
    require_binding_match format OURS_VIT_REORDER_BINDING_V1
    require_binding_match status PASS
    require_binding_match model_sha256 \
      "$(sha256sum "$model" | awk '{print $1}')"
    require_binding_match input_sha256 \
      "$(sha256sum "$input" | awk '{print $1}')"
    require_binding_match plan_sha256 \
      "$(sha256sum "$head_plan" | awk '{print $1}')"
    require_binding_match block_limit "$stop_after_block"
    reorder_report=$(binding_value oracle_report)
    if [[ ! -f "$reorder_report" ]]; then
      echo "missing bound reorder report: $reorder_report" >&2
      exit 2
    fi
    require_binding_match oracle_report_sha256 \
      "$(sha256sum "$reorder_report" | awk '{print $1}')"
    echo "OURS_HEAD_REORDER_PREVALIDATION status=PASS binding=$prevalidated_reorder_binding"
  else
    python_bin=${PYTHON_BIN:-python3}
    "$python_bin" "$root/ours_vit_b16/scripts/verify_head_reorder.py" \
      --model "$model" --input "$input" --head-plan "$head_plan" \
      --output "$output_dir/head_reorder_oracle.json" \
      --block-limit "$stop_after_block" \
      > "$output_dir/head_reorder_oracle.log" 2>&1
  fi
fi
alice_pid=
bob_pid=
qdisc_owned=0
cleanup() {
  if [[ -n "$alice_pid" ]]; then kill -TERM -- "-$alice_pid" 2>/dev/null || true; fi
  if [[ -n "$bob_pid" ]]; then kill -TERM -- "-$bob_pid" 2>/dev/null || true; fi
  if [[ "$qdisc_owned" == 1 ]]; then
    tc qdisc del dev lo root 2>/dev/null || true
  fi
  if [[ "$network_mode" == netem ]]; then
    tc qdisc show dev lo > "$output_dir/qdisc_after.txt" 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$network_mode" == netem ]]; then
  tc qdisc show dev lo > "$output_dir/qdisc_before.txt"
  if ! grep -q '^qdisc noqueue ' "$output_dir/qdisc_before.txt"; then
    echo "refusing to replace a pre-existing loopback qdisc" >&2
    exit 3
  fi
  # 0.3 ms RTT means 0.15 ms delay in each loopback direction.
  tc qdisc replace dev lo root netem limit 100000 delay 0.15ms rate 1gbit
  qdisc_owned=1
  tc qdisc show dev lo > "$output_dir/qdisc_active.txt"
else
  printf '%s\n' 'network shaping disabled; diagnostic/smoke only' \
    > "$output_dir/qdisc_active.txt"
fi

{
  date --iso-8601=ns
  echo "backend=$backend"
  echo "binary=$binary"
  echo "binary_sha256=$(sha256sum "$binary" | awk '{print $1}')"
  echo "model=$model"
  echo "model_sha256=$(sha256sum "$model" | awk '{print $1}')"
  echo "input=$input"
  echo "input_sha256=$(sha256sum "$input" | awk '{print $1}')"
  echo "logical_tokens=197"
  echo "physical_tokens=256"
  echo "transformer_blocks=$stop_after_block"
  echo "ablation=${ablation^^}"
  echo "head_plan=${head_plan:-built_in_all_softmax_original_order}"
  if [[ -n "${OURS_VIT_STOP_AFTER_NORMALIZATION:-}" ]]; then
    echo "cost_scope=a2_normalization_only"
    echo "conversion_cost_included=0"
    echo "conversion_cost_deferred_to=A3"
  fi
  if [[ "$network_mode" == netem ]]; then
    echo "network=1Gbps/0.3ms_RTT"
  else
    echo "network=unshaped_diagnostic_only"
  fi
  echo "alice_cpus=0-47,96-111"
  echo "bob_cpus=48-95,144-159"
  echo "omp_threads_per_party=$omp_threads"
  echo "mpc_workers_per_party=64"
  if [[ "$backend" == gpu ]]; then
    echo "alice_gpu_id=$alice_gpu_id"
    echo "bob_gpu_id=$bob_gpu_id"
    echo "gpu_direct_plaintext=$gpu_direct_plaintext"
    echo "gpu_linear_streams=$gpu_linear_streams"
    # Some managed hosts reject the per-device name/UUID query even though
    # CUDA and the basic all-device telemetry interface are healthy. Record
    # only the fields needed to prove the selected device was idle enough.
    nvidia-smi \
      --query-gpu=index,utilization.gpu,memory.used,memory.free \
      --format=csv,noheader || \
      echo "gpu_telemetry=unavailable; runtime CUDA/Phantom markers required"
  fi
  lscpu | grep -E 'Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core'
} > "$output_dir/environment.txt"

alice_env=(env OMP_NUM_THREADS="$omp_threads" OMP_DYNAMIC=FALSE)
bob_env=(env OMP_NUM_THREADS="$omp_threads" OMP_DYNAMIC=FALSE)
if [[ "$backend" == gpu ]]; then
  alice_env+=(CUDA_VISIBLE_DEVICES="$alice_gpu_id"
    BLB_VIT_GPU_DIRECT_PLAINTEXT="$gpu_direct_plaintext"
    BLB_VIT_GPU_LINEAR_STREAMS="$gpu_linear_streams")
  bob_env+=(CUDA_VISIBLE_DEVICES="$bob_gpu_id"
    BLB_VIT_GPU_DIRECT_PLAINTEXT="$gpu_direct_plaintext"
    BLB_VIT_GPU_LINEAR_STREAMS="$gpu_linear_streams")
fi

ours_args=(--ablation "$ablation")
if [[ -n "$head_plan" ]]; then ours_args+=(--head-plan "$head_plan"); fi

date --iso-8601=ns > "$output_dir/start.txt"
setsid /usr/bin/time -v -o "$output_dir/alice.resource.txt" \
  timeout --foreground --signal=TERM --kill-after=30s "${timeout_s}s" \
  taskset -c 0-47,96-111 "${alice_env[@]}" \
  "$binary" --party 1 --port "$port" --address 127.0.0.1 \
  --model "$model" --input "$input" --output "$output_dir" \
  --backend "$backend" --stop-after-block "$stop_after_block" \
  "${ours_args[@]}" \
  > "$output_dir/alice.log" 2>&1 &
alice_pid=$!

sleep 1
setsid /usr/bin/time -v -o "$output_dir/bob.resource.txt" \
  timeout --foreground --signal=TERM --kill-after=30s "${timeout_s}s" \
  taskset -c 48-95,144-159 "${bob_env[@]}" \
  "$binary" --party 2 --port "$port" --address 127.0.0.1 \
  --model "$model" --input "$input" --output "$output_dir" \
  --backend "$backend" --stop-after-block "$stop_after_block" \
  "${ours_args[@]}" \
  > "$output_dir/bob.log" 2>&1 &
bob_pid=$!

set +e
wait "$alice_pid"; alice_rc=$?; alice_pid=
if [[ $alice_rc -ne 0 && -n "$bob_pid" ]]; then kill -TERM -- "-$bob_pid" 2>/dev/null || true; fi
wait "$bob_pid"; bob_rc=$?; bob_pid=
set -e

if [[ "$network_mode" == netem ]]; then
  tc -s qdisc show dev lo > "$output_dir/qdisc_end.txt"
fi
date --iso-8601=ns > "$output_dir/end.txt"
printf 'alice_rc=%s\nbob_rc=%s\n' "$alice_rc" "$bob_rc" > "$output_dir/status.txt"
if [[ $alice_rc -ne 0 || $bob_rc -ne 0 ]]; then exit 1; fi

# Tensor/noise diagnostics intentionally stop before the profiler writes the
# end-to-end latency CSV files.  The two protocol parties have already exited
# successfully, so do not misreport the expected absence of those CSV files as
# a failed secure run.
if [[ -n "${BLB_QKV_AGGREGATE:-}" ||
      -n "${OURS_VIT_A0_TRACE:-}" ||
      -n "${OURS_VIT_PARTIAL_FINAL_LOGITS:-}" ||
      -n "${OURS_VIT_A0_PARTIAL_FINAL_LOGITS:-}" ||
      -n "${OURS_VIT_STOP_AFTER_NORMALIZATION:-}" ||
      -n "${OURS_VIT_STOP_AFTER_ATTENTION:-}" ||
      -n "${BLB_VIT_DIAGNOSTIC_STOP_AFTER_QK:-}" ||
      -n "${BLB_VIT_DIAGNOSTIC_STOP_AFTER_SOFTMAX:-}" ||
      -n "${BLB_VIT_DIAGNOSTIC_SOFTMAX_V_PATH:-}" ]]; then
  echo "OURS_VIT_RUN status=PASS_DIAGNOSTIC backend=$backend output=$output_dir"
  exit 0
fi

python_bin=${PYTHON_BIN:-python3}
summarizer="$root/vit_b16/scripts/summarize_blb_vit_baseline.py"
if [[ -f "$summarizer" ]]; then
  "$python_bin" "$summarizer" --run-dir "$output_dir"
  echo "OURS_VIT_RUN status=PASS backend=$backend output=$output_dir"
else
  # The minimal `ours` image intentionally does not copy the BLB source-tree
  # symlink target.  A host wrapper runs the same read-only summarizer against
  # the mounted results after both secure parties have exited successfully.
  echo "OURS_VIT_RUN status=PASS_SUMMARY_DEFERRED backend=$backend output=$output_dir"
fi
