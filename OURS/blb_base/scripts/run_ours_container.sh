#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 B|H|HA|HP|HAP|J|BP HEAD_PLAN OUTPUT_DIR BLOCKS [attention|block|logits|full]" >&2
  exit 2
fi

version=$1
head_plan=$2
output_dir=$3
blocks=$4
scope=${5:-attention}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
image=${OURS_DOCKER_IMAGE:-ours:restored-fast-20260903d}
port=${OURS_VIT_PORT:-31003}
backend=${OURS_VIT_BACKEND:-gpu}
network_mode=${OURS_VIT_NETWORK_MODE:-netem}
min_free_memory=${OURS_VIT_MIN_FREE_MEMORY_MIB:-70000}
max_gpu_utilization=${OURS_VIT_MAX_GPU_UTILIZATION_PERCENT:-5}
if [[ "$backend" != cpu && "$backend" != gpu ]]; then
  echo "OURS_VIT_BACKEND must be cpu or gpu" >&2
  exit 2
fi
gpu_direct_plaintext=${BLB_VIT_GPU_DIRECT_PLAINTEXT:-1}
gpu_linear_streams=${BLB_VIT_GPU_LINEAR_STREAMS:-8}
allow_busy_gpus=${OURS_VIT_ALLOW_BUSY_GPUS:-0}
allow_shared_gpu=${OURS_VIT_ALLOW_SHARED_GPU:-0}
binary_override=${OURS_VIT_BINARY_OVERRIDE:-$root/build/container-gpu/bin/ours_vit_b16_main}
phantom_library_override=${OURS_VIT_PHANTOM_LIBRARY_OVERRIDE:-$root/build/container-gpu/extern/phantom/lib/libPhantom.so}
cuda_gdb_command_file=${BLB_VIT_CUDA_GDB_COMMAND_FILE:-}

case "$version" in
  B|H|HA) packing=0 ;;
  HP|HAP|J|BP) packing=1 ;;
  *) echo "version must be B,H,HA,HP,HAP,J,BP" >&2; exit 2 ;;
esac
if [[ "$scope" != attention && "$scope" != block &&
      "$scope" != logits && "$scope" != full ]]; then
  echo "scope must be attention, block, logits, or full" >&2
  exit 2
fi
if [[ ! "$blocks" =~ ^([1-9]|1[0-2])$ ]]; then
  echo "BLOCKS must be an integer in [1, 12]" >&2
  exit 2
fi
if [[ "$scope" == full && "$blocks" != 12 ]]; then
  echo "full scope requires BLOCKS=12; use logits for a classified prefix" >&2
  exit 2
fi
if [[ "$gpu_direct_plaintext" != 0 && "$gpu_direct_plaintext" != 1 ]]; then
  echo "BLB_VIT_GPU_DIRECT_PLAINTEXT must be 0 or 1" >&2
  exit 2
fi
if [[ ! "$gpu_linear_streams" =~ ^([1-9]|1[0-6])$ ]]; then
  echo "BLB_VIT_GPU_LINEAR_STREAMS must be an integer in [1, 16]" >&2
  exit 2
fi
if [[ "$allow_busy_gpus" != 0 && "$allow_busy_gpus" != 1 ]]; then
  echo "OURS_VIT_ALLOW_BUSY_GPUS must be 0 or 1" >&2
  exit 2
fi
if [[ "$allow_shared_gpu" != 0 && "$allow_shared_gpu" != 1 ]]; then
  echo "OURS_VIT_ALLOW_SHARED_GPU must be 0 or 1" >&2
  exit 2
fi
for override in "$binary_override" "$phantom_library_override"; do
  if [[ -n "$override" && ("$override" != /* || ! -f "$override") ]]; then
    echo "binary/library override must be an existing absolute file" >&2
    exit 2
  fi
done
if [[ -n "$binary_override" && ! -x "$binary_override" ]]; then
  echo "OURS_VIT_BINARY_OVERRIDE is not executable: $binary_override" >&2
  exit 2
fi
if [[ -n "$cuda_gdb_command_file" ]]; then
  if [[ "$cuda_gdb_command_file" != /* || ! -f "$cuda_gdb_command_file" ]]; then
    echo "BLB_VIT_CUDA_GDB_COMMAND_FILE must be an existing absolute path" >&2
    exit 2
  fi
fi
docker_extra_mounts=()
if [[ -n "$cuda_gdb_command_file" ]]; then
  docker_extra_mounts+=(
    --mount
    "type=bind,src=$cuda_gdb_command_file,dst=/workspace/OURS/native_mismatch_cuda_gdb.cmd,readonly"
  )
fi
if [[ -n "$binary_override" ]]; then
  docker_extra_mounts+=(
    --mount
    "type=bind,src=$binary_override,dst=/workspace/OURS/build/gpu/bin/ours_vit_b16_main,readonly"
  )
fi
if [[ -n "$phantom_library_override" ]]; then
  docker_extra_mounts+=(
    --mount
    "type=bind,src=$phantom_library_override,dst=/opt/ours/lib/libPhantom.so,readonly"
  )
fi
if [[ "$output_dir" != outputs/* ||
      "/$output_dir/" == *"/../"* ]]; then
  echo "OUTPUT_DIR must be below outputs/" >&2
  exit 2
fi
if [[ "$head_plan" != /* ]]; then head_plan="$root/$head_plan"; fi
if [[ ! -f "$head_plan" ]]; then
  echo "missing head plan: $head_plan" >&2
  exit 2
fi
if [[ "$image" != ours && "$image" != ours:* ]]; then
  echo "OURS_DOCKER_IMAGE must be ours or an ours:<tag> name" >&2
  exit 2
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "missing image $image; run scripts/build_ours_image.sh first" >&2
  exit 2
fi

host_output_dir="$root/$output_dir"
if [[ -e "$host_output_dir" ]]; then
  echo "refusing to reuse existing output directory: $host_output_dir" >&2
  exit 2
fi
mkdir -p "$host_output_dir"
checkpoint_dir=$(readlink -f "$root/checkpoints/vit_b16")
host_model=${OURS_VIT_MODEL:-$checkpoint_dir/bolt_model.txt}
host_model=$(readlink -f "$host_model")
input_dir=$(readlink -f "$root/inputs/vit_b16")
host_input=${OURS_VIT_INPUT:-$input_dir/bob_image.txt}
host_input=$(readlink -f "$host_input")
if [[ ! -f "$host_input" ]]; then
  echo "missing preprocessed ViT input: $host_input" >&2
  exit 2
fi
image_id=$(docker image inspect "$image" --format '{{.Id}}')

# Run the independent FP32/fixed-clear head-reorder oracle on the host.  The
# resulting hash binding is checked by both secure parties before sockets open.
python3 -B "$root/ours_vit_b16/scripts/verify_head_reorder.py" \
  --model "$host_model" \
  --input "$host_input" \
  --head-plan "$head_plan" \
  --output "$host_output_dir/head_reorder_oracle.json" \
  --block-limit "$blocks" > "$host_output_dir/head_reorder_oracle.log" 2>&1

container_plan=/workspace/OURS/runtime_head_plan.txt
container_report="$output_dir/head_reorder_oracle.json"
binding="$host_output_dir/reorder_binding.txt"
{
  echo "format=OURS_VIT_REORDER_BINDING_V1"
  echo "status=PASS"
  echo "model_sha256=$(sha256sum "$host_model" | awk '{print $1}')"
  echo "input_sha256=$(sha256sum "$host_input" | awk '{print $1}')"
  echo "attention_packing=$([[ "$packing" == 1 ]] && echo packed || echo unpacked)"
  echo "plan_sha256=$(sha256sum "$head_plan" | awk '{print $1}')"
  echo "block_limit=$blocks"
  echo "oracle_report=$container_report"
  echo "oracle_report_sha256=$(sha256sum "$host_output_dir/head_reorder_oracle.json" | awk '{print $1}')"
} > "$binding"

# B is the immutable built-in all-Softmax baseline and deliberately rejects
# a runtime --head-plan.  Keep the externally supplied plan for the host-side
# reference/oracle and provenance, but do not bind it into the B binary.
runtime_head_plan="$container_plan"
if [[ "$version" == B ]]; then
  runtime_head_plan=""
fi

# CPU execution exposes no GPU. GPU execution honors the configured pair.
docker_device_args=()
gpu_csv=""
container_alice_gpu_id=""
container_bob_gpu_id=""
container_visible_gpus=""
party_topology=cpu
if [[ "$backend" == gpu ]]; then
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,utilization.gpu,memory.used,memory.free \
  --format=csv,noheader > "$host_output_dir/gpu_telemetry_before_run.txt"
if [[ -n "${OURS_VIT_PHYSICAL_GPUS:-}" ]]; then
  if [[ ! "$OURS_VIT_PHYSICAL_GPUS" =~ ^[0-9]+,[0-9]+$ ]]; then
    echo "OURS_VIT_PHYSICAL_GPUS must be two comma-separated GPU indices" >&2
    exit 2
  fi
  IFS=, read -r -a selected_gpus <<< "$OURS_VIT_PHYSICAL_GPUS"
  mapfile -t currently_eligible < <(
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free \
      --format=csv,noheader,nounits |
    awk -F, -v allow_busy="$allow_busy_gpus" -v min_free="$min_free_memory" -v max_util="$max_gpu_utilization" \
      '($4 + 0 >= min_free && (allow_busy == 1 || $2 + 0 <= max_util)) {
         print $1 + 0
       }'
  )
  for selected_gpu in "${selected_gpus[@]}"; do
    eligible=0
    for candidate_gpu in "${currently_eligible[@]}"; do
      [[ "$selected_gpu" == "$candidate_gpu" ]] && eligible=1
    done
    if [[ "$eligible" != 1 ]]; then
      echo "fixed cohort GPU $selected_gpu busy: requires utilization <=${max_gpu_utilization}% (unless allow_busy=1) and free memory >=${min_free_memory} MiB; no secure run started" >&2
      exit 75
    fi
  done
else
  mapfile -t selected_gpus < <(
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free \
      --format=csv,noheader,nounits |
    awk -F, -v allow_busy="$allow_busy_gpus" -v min_free="$min_free_memory" -v max_util="$max_gpu_utilization" \
      '$4 + 0 >= min_free && (allow_busy == 1 || $2 + 0 <= max_util) {
                print $1 + 0, $3 + 0, $4 + 0
              }' |
    sort -k2,2n -k3,3nr |
    awk 'NR <= 2 {print $1}'
  )
fi
if [[ ${#selected_gpus[@]} -ne 2 ]]; then
  echo "need two selected GPUs (memory.free >=70000 MiB)" >&2
  exit 3
fi
if [[ "${selected_gpus[0]}" == "${selected_gpus[1]}" &&
      "$allow_shared_gpu" != 1 ]]; then
  echo "shared physical GPU requires OURS_VIT_ALLOW_SHARED_GPU=1" >&2
  exit 3
fi
gpu_csv=${selected_gpus[0]},${selected_gpus[1]}
if [[ "${selected_gpus[0]}" == "${selected_gpus[1]}" ]]; then
  gpu_request="device=${selected_gpus[0]}"
  party_topology=shared_physical_gpu
  container_visible_gpus=0,0
else
  gpu_request="\"device=$gpu_csv\""
  party_topology=separate_physical_gpus
  container_visible_gpus=0,1
fi
gpu_uuid_for_index() {
  awk -F, -v target="$1" '
    function trim(value) {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      return value
    }
    trim($1) + 0 == target { print trim($3); exit }
  ' "$host_output_dir/gpu_telemetry_before_run.txt"
}
container_alice_gpu_id=$(gpu_uuid_for_index "${selected_gpus[0]}")
container_bob_gpu_id=$(gpu_uuid_for_index "${selected_gpus[1]}")
if [[ -z "$container_alice_gpu_id" || -z "$container_bob_gpu_id" ]]; then
  echo "cannot resolve selected GPU UUIDs from NVIDIA telemetry" >&2
  exit 3
fi
docker_device_args+=(--gpus "$gpu_request")
else
  docker_device_args+=(--env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES=)
fi
container_name="ours-${version}-${scope}-$(date +%Y%m%d-%H%M%S)-$$"
{
  echo "selection=$([[ -n "${OURS_VIT_PHYSICAL_GPUS:-}" ]] && echo fixed_by_paired_version || echo dynamic_immediately_before_container_start)"
  echo "backend=$backend"
  echo "physical_gpus=$gpu_csv"
  echo "physical_gpu_uuids=$container_alice_gpu_id,$container_bob_gpu_id"
  echo "container_cuda_devices=$container_visible_gpus"
  echo "party_topology=$party_topology"
  if [[ "$allow_busy_gpus" == 1 ]]; then
    echo "eligibility=explicit_busy_gpu_override_memory_free_ge_${min_free_memory}_MiB"
  else
    echo "eligibility=utilization_le_${max_gpu_utilization}_percent_and_memory_free_ge_${min_free_memory}_MiB"
  fi
  echo "allow_busy_gpus=$allow_busy_gpus"
  echo "image=$image"
  echo "image_id=$image_id"
  echo "plan=$head_plan"
  echo "plan_sha256=$(sha256sum "$head_plan" | awk '{print $1}')"
  echo "input=$host_input"
  echo "input_sha256=$(sha256sum "$host_input" | awk '{print $1}')"
  echo "conversion_path=see_edge_cloud_contract_and_run_metadata"
  echo "gpu_direct_plaintext=$gpu_direct_plaintext"
  echo "gpu_linear_streams=$gpu_linear_streams"
  echo "attention_diagonal_kernel=complete_v1"
  echo "allow_shared_gpu=$allow_shared_gpu"
  echo "binary_override=${binary_override:-image_payload}"
  echo "phantom_library_override=${phantom_library_override:-image_payload}"
  if [[ -n "$binary_override" ]]; then
    echo "binary_override_sha256=$(sha256sum "$binary_override" | awk '{print $1}')"
  fi
  date --iso-8601=ns
} > "$host_output_dir/gpu_selection.txt"

echo "OURS_GPU_SELECTION status=PASS physical_gpus=$gpu_csv"
echo "OURS_CONTAINER name=$container_name image=$image image_id=$image_id scope=$scope"
diagnostic_env=()
timeout_seconds=1200
if [[ "$scope" == attention ]]; then
  diagnostic_env+=(OURS_VIT_STOP_AFTER_ATTENTION=1)
  timeout_seconds=900
elif [[ "$scope" == logits ]]; then
  diagnostic_env+=(OURS_VIT_PARTIAL_FINAL_LOGITS=1)
  timeout_seconds=1200
elif [[ "$scope" == full ]]; then
  # A real 12-block run can exceed the historical 20-minute diagnostic
  # ceiling, especially for All-Softmax. Keep a finite fail-safe without
  # truncating a valid paper run. Callers may still override it explicitly.
  timeout_seconds=21600
fi
timeout_seconds=${BLB_VIT_TIMEOUT_SECONDS:-$timeout_seconds}
if [[ "${OURS_VIT_CAPTURE_ATTENTION_TRACE:-0}" == 1 ]]; then
  if [[ ("$scope" != attention && "$scope" != logits) ||
        ("$blocks" != 1 && !("$version" == B && "$scope" == logits && "$blocks" == 2)) ]]; then
    echo "OURS_VIT_CAPTURE_ATTENTION_TRACE requires a one-block diagnostic or B two-block logits" >&2
    exit 2
  fi
  if [[ "$version" == B ]]; then
    diagnostic_env+=(OURS_VIT_B_TRACE=1 OURS_VIT_B_TRACE_DIR="$output_dir")
  else
    diagnostic_env+=(
      OURS_VIT_STOP_AFTER_ATTENTION=1
      OURS_VIT_HETERO_ATTENTION_TRACE=1
      OURS_VIT_HETERO_ATTENTION_TRACE_DIR="$output_dir"
    )
  fi
  echo "OURS_DIAGNOSTIC_TRACE enabled=1 feeds_secure_graph=0 timing_eligible=0"
fi

# Run the immutable ID captured before GPU selection.  The mutable
# ours:latest tag may be rebuilt by another experiment while this runner is
# preparing the host-side reorder oracle.
docker run --rm --pull=never --name "$container_name" \
  --network=none --cap-add=NET_ADMIN --ipc=host \
  "${docker_device_args[@]}" \
  --mount "type=bind,src=$host_output_dir,dst=/workspace/OURS/$output_dir" \
  --mount "type=bind,src=$checkpoint_dir,dst=/workspace/OURS/checkpoints/vit_b16,readonly" \
  --mount "type=bind,src=$host_model,dst=/workspace/OURS/runtime_model.txt,readonly" \
  --mount "type=bind,src=$input_dir,dst=/workspace/OURS/inputs/vit_b16,readonly" \
  --mount "type=bind,src=$host_input,dst=/workspace/OURS/runtime_input.txt,readonly" \
  --mount "type=bind,src=$head_plan,dst=$container_plan,readonly" \
  --mount "type=bind,src=$root/scripts/run_ours_vit_b16.sh,dst=/workspace/OURS/scripts/run_ours_vit_b16.sh,readonly" \
  "${docker_extra_mounts[@]}" \
  --mount "type=bind,src=/usr/local/cuda-12.3,dst=/usr/local/cuda-12.3,readonly" \
  --workdir /workspace/OURS "$image_id" env \
  OURS_VIT_BINARY_OVERRIDE=/workspace/OURS/build/gpu/bin/ours_vit_b16_main \
  OURS_VIT_MATCH_LINEAR_HE_CONTROL="${OURS_VIT_MATCH_LINEAR_HE_CONTROL:-0}" \
  OURS_VIT_FORCE_GENERIC_ALL_S="${OURS_VIT_FORCE_GENERIC_ALL_S:-0}" \
  OURS_VIT_MPC_WORKERS="${OURS_VIT_MPC_WORKERS:-64}" \
  OURS_VIT_C2M_HEADROOM_BITS="${OURS_VIT_C2M_HEADROOM_BITS:-24}" \
  OURS_VIT_VERSION="$version" \
  OURS_VIT_HEAD_PLAN="$runtime_head_plan" \
  BLB_VIT_GPU_DIRECT_PLAINTEXT="$gpu_direct_plaintext" \
  BLB_VIT_GPU_LINEAR_STREAMS="$gpu_linear_streams" \
  BLB_VIT_OMP_THREADS="${BLB_VIT_OMP_THREADS:-64}" \
  BLB_VIT_CUDA_GDB_COMMAND_FILE="${cuda_gdb_command_file:+/workspace/OURS/native_mismatch_cuda_gdb.cmd}" \
  OURS_VIT_PREVALIDATED_REORDER_BINDING="$output_dir/reorder_binding.txt" \
  BLB_VIT_INPUT=/workspace/OURS/runtime_input.txt \
  BLB_VIT_MODEL=/workspace/OURS/runtime_model.txt \
  OURS_VIT_NETWORK_MODE="$network_mode" \
  BLB_VIT_STOP_AFTER_BLOCK="$blocks" \
  BLB_VIT_TIMEOUT_SECONDS="$timeout_seconds" \
  "${diagnostic_env[@]}" \
  scripts/run_ours_vit_b16.sh "$backend" "$output_dir" "$port" \
    "$container_alice_gpu_id" "$container_bob_gpu_id"

# The inner BLB runner writes its own GPU audit to the same filename. Append
# the immutable outer binding after a successful container exit so collectors
# can prove exactly which image and fixed plan produced the measurements.
{
  if [[ -n "${OURS_VIT_PHYSICAL_GPUS:-}" ]]; then
    echo "outer_selection=fixed_external_pair"
  else
    echo "outer_selection=dynamic_immediately_before_container_start"
  fi
  echo "backend=$backend"
  echo "physical_gpus=$gpu_csv"
  echo "physical_gpu_uuids=$container_alice_gpu_id,$container_bob_gpu_id"
  echo "container_visible_gpus=$container_visible_gpus"
  echo "party_topology=$party_topology"
  echo "image=$image"
  echo "image_id=$image_id"
  echo "plan=$head_plan"
  echo "plan_sha256=$(sha256sum "$head_plan" | awk '{print $1}')"
  echo "attention_packing=$([[ "$packing" == 1 ]] && echo packed || echo unpacked)"
  echo "allow_busy_gpus=$allow_busy_gpus"
  echo "gpu_direct_plaintext=$gpu_direct_plaintext"
  echo "gpu_linear_streams=$gpu_linear_streams"
  echo "attention_diagonal_kernel=complete_v1"
  echo "allow_shared_gpu=$allow_shared_gpu"
  echo "binary_override=${binary_override:-image_payload}"
  echo "phantom_library_override=${phantom_library_override:-image_payload}"
  if [[ -n "$binary_override" ]]; then
    echo "binary_override_sha256=$(sha256sum "$binary_override" | awk '{print $1}')"
  fi
} >> "$host_output_dir/gpu_selection.txt"

# Stage-specific collectors own result validation. An obsolete assignment-cost
# schema must not turn a successful two-party run into an execution failure.

echo "OURS_HETERO_CONTAINER_RUN status=PASS version=${version^^} scope=$scope output=$output_dir"
