#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 outputs/paper/a0/RUN_NAME" >&2
  exit 2
fi

output_dir=$1
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
image=${OURS_DOCKER_IMAGE:-ours}
port=${OURS_VIT_PORT:-30901}

if [[ "$output_dir" != outputs/paper/a0/* ]]; then
  echo "OUTPUT_DIR must be below outputs/paper/a0" >&2
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
input_dir=$(readlink -f "$root/inputs/vit_b16")
host_input=${OURS_VIT_INPUT:-$input_dir/bob_image.txt}
host_input=$(readlink -f "$host_input")
if [[ ! -f "$host_input" ]]; then
  echo "missing preprocessed ViT input: $host_input" >&2
  exit 2
fi
outputs_dir=$(readlink -f "$root/outputs")
image_id=$(docker image inspect "$image" --format '{{.Id}}')

nvidia-smi \
  --query-gpu=index,name,uuid,driver_version,memory.total,utilization.gpu,memory.used,memory.free \
  --format=csv,noheader > "$host_output_dir/gpu_telemetry_before_run.txt"
if [[ -n "${OURS_VIT_PHYSICAL_GPUS:-}" ]]; then
  IFS=, read -r -a selected_gpus <<< "$OURS_VIT_PHYSICAL_GPUS"
  selection_mode=explicit
else
  mapfile -t selected_gpus < <(
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free \
      --format=csv,noheader,nounits |
    awk -F, '$2 + 0 <= 5 && $4 + 0 >= 70000 {
                print $1 + 0, $3 + 0, $4 + 0
              }' |
    sort -k2,2n -k3,3nr |
    awk 'NR <= 2 {print $1}'
  )
  selection_mode=dynamic_immediately_before_container_start
fi
if [[ ${#selected_gpus[@]} -ne 2 ||
      ! "${selected_gpus[0]}" =~ ^[0-9]+$ ||
      ! "${selected_gpus[1]}" =~ ^[0-9]+$ ||
      "${selected_gpus[0]}" == "${selected_gpus[1]}" ]]; then
  echo "need two distinct idle GPUs (utilization <=5%, memory.free >=70000 MiB)" >&2
  exit 3
fi

gpu_csv=${selected_gpus[0]},${selected_gpus[1]}
gpu_request="\"device=$gpu_csv\""
container_name="ours-a0-trace-$(date +%Y%m%d-%H%M%S)-$$"
{
  echo "selection=$selection_mode"
  echo "physical_gpus=$gpu_csv"
  echo "container_cuda_devices=0,1"
  echo "party_topology=separate_physical_gpus"
  echo "image=$image"
  echo "image_id=$image_id"
  echo "input=$host_input"
  echo "input_sha256=$(sha256sum "$host_input" | awk '{print $1}')"
  date --iso-8601=ns
} > "$host_output_dir/gpu_selection.txt"

echo "OURS_GPU_SELECTION status=PASS physical_gpus=$gpu_csv"
echo "OURS_A0_CONTAINER name=$container_name image_id=$image_id scope=full_trace"
docker run --rm --pull=never --name "$container_name" \
  --network=none --cap-add=NET_ADMIN --ipc=host \
  --gpus "$gpu_request" \
  --mount "type=bind,src=$outputs_dir,dst=/workspace/OURS/outputs" \
  --mount "type=bind,src=$checkpoint_dir,dst=/workspace/OURS/checkpoints/vit_b16,readonly" \
  --mount "type=bind,src=$host_input,dst=/workspace/OURS/runtime_input.txt,readonly" \
  --mount "type=bind,src=/usr/local/cuda-12.3,dst=/usr/local/cuda-12.3,readonly" \
  --workdir /workspace/OURS "$image_id" env \
  OURS_VIT_ABLATION=a0 \
  OURS_VIT_NETWORK_MODE=netem \
  OURS_VIT_A0_TRACE=1 \
  OURS_VIT_A0_TRACE_DIR="$output_dir" \
  OURS_VIT_A0_PARTIAL_FINAL_LOGITS=1 \
  BLB_VIT_INPUT=/workspace/OURS/runtime_input.txt \
  BLB_VIT_STOP_AFTER_BLOCK=1 \
  BLB_VIT_TIMEOUT_SECONDS="${BLB_VIT_TIMEOUT_SECONDS:-3600}" \
  scripts/run_ours_vit_b16.sh gpu "$output_dir" "$port" 0 1

# The inner runner records the container-visible GPU IDs. Preserve the outer
# physical-device and immutable-image binding after successful completion.
{
  echo "outer_physical_gpus=$gpu_csv"
  echo "outer_image=$image"
  echo "outer_image_id=$image_id"
  echo "outer_input=$host_input"
  echo "outer_input_sha256=$(sha256sum "$host_input" | awk '{print $1}')"
} >> "$host_output_dir/gpu_selection.txt"

gate_args=()
if [[ "${OURS_VIT_ALLOW_INCOMPLETE_A0:-0}" != 1 ]]; then
  gate_args+=(--require-complete-gate)
fi
python3 -B "$root/ours_vit_b16/scripts/diagnostics/compare_a0_trace.py" \
  --model "$checkpoint_dir/bolt_model.txt" \
  --input "$host_input" \
  --trace-dir "$host_output_dir" \
  --secure-logits "$host_output_dir/secure_logits_party1.txt" \
  --secure-prediction "$host_output_dir/prediction_party1.txt" \
  --output "$host_output_dir/a0_trace_comparison.json" \
  "${gate_args[@]}"

echo "OURS_A0_CONTAINER_RUN status=PASS output=$output_dir"
