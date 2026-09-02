#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 a3|a4 HEAD_PLAN OUTPUT_DIR BLOCKS [attention|block|logits|full]" >&2
  exit 2
fi

ablation=${1,,}
head_plan=$2
output_dir=$3
blocks=$4
scope=${5:-attention}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
image=${OURS_DOCKER_IMAGE:-ours}
port=${OURS_VIT_PORT:-31003}

if [[ "$ablation" == a2 &&
      "${OURS_VIT_DIAGNOSTIC_A2_BASELINE:-0}" != 1 ]]; then
  echo "a2 is accepted only with OURS_VIT_DIAGNOSTIC_A2_BASELINE=1" >&2
  exit 2
fi
if [[ "$ablation" != a2 && "$ablation" != a3 && "$ablation" != a4 ]]; then
  echo "ablation must be a3 or a4" >&2
  exit 2
fi
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
if [[ "$output_dir" != outputs/a3_a4_review/* ]]; then
  echo "OUTPUT_DIR must be below outputs/a3_a4_review" >&2
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
input_dir=$(readlink -f "$root/inputs/vit_b16")
review_dir=$(readlink -f "$root/outputs/a3_a4_review")
image_id=$(docker image inspect "$image" --format '{{.Id}}')

# Run the independent FP32/fixed-clear head-reorder oracle on the host.  The
# resulting hash binding is checked by both secure parties before sockets open.
python3 -B "$root/ours_vit_b16/scripts/verify_head_reorder.py" \
  --model "$checkpoint_dir/bolt_model.txt" \
  --input "$input_dir/bob_image.txt" \
  --head-plan "$head_plan" \
  --output "$host_output_dir/head_reorder_oracle.json" \
  --block-limit "$blocks" > "$host_output_dir/head_reorder_oracle.log" 2>&1

container_plan=/workspace/OURS/runtime_head_plan.txt
container_report="$output_dir/head_reorder_oracle.json"
binding="$host_output_dir/reorder_binding.txt"
{
  echo "format=OURS_VIT_REORDER_BINDING_V1"
  echo "status=PASS"
  echo "model_sha256=$(sha256sum "$checkpoint_dir/bolt_model.txt" | awk '{print $1}')"
  echo "input_sha256=$(sha256sum "$input_dir/bob_image.txt" | awk '{print $1}')"
  echo "plan_sha256=$(sha256sum "$head_plan" | awk '{print $1}')"
  echo "block_limit=$blocks"
  echo "oracle_report=$container_report"
  echo "oracle_report_sha256=$(sha256sum "$host_output_dir/head_reorder_oracle.json" | awk '{print $1}')"
} > "$binding"

# Select immediately before docker run.  Two distinct GPUs are required so the
# result remains comparable to the BLB two-party/two-GPU baseline.  Prefer the
# least allocated cards; no physical index is embedded in code or the image.
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free \
  --format=csv,noheader > "$host_output_dir/gpu_telemetry_before_run.txt"
mapfile -t selected_gpus < <(
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free \
    --format=csv,noheader,nounits |
  awk -F, '$2 + 0 <= 5 && $4 + 0 >= 70000 {
              print $1 + 0, $3 + 0, $4 + 0
            }' |
  sort -k2,2n -k3,3nr |
  awk 'NR <= 2 {print $1}'
)
if [[ ${#selected_gpus[@]} -ne 2 ||
      "${selected_gpus[0]}" == "${selected_gpus[1]}" ]]; then
  echo "need two distinct idle GPUs (utilization <=5%, memory.free >=70000 MiB)" >&2
  exit 3
fi
gpu_csv=${selected_gpus[0]},${selected_gpus[1]}
gpu_request="\"device=$gpu_csv\""
container_name="ours-${ablation}-${scope}-$(date +%Y%m%d-%H%M%S)-$$"
{
  echo "selection=dynamic_immediately_before_container_start"
  echo "physical_gpus=$gpu_csv"
  echo "container_cuda_devices=0,1"
  echo "party_topology=separate_physical_gpus"
  echo "eligibility=utilization_le_5_percent_and_memory_free_ge_70000_MiB"
  echo "image=$image"
  echo "image_id=$image_id"
  echo "plan=$head_plan"
  echo "plan_sha256=$(sha256sum "$head_plan" | awk '{print $1}')"
  date --iso-8601=ns
} > "$host_output_dir/gpu_selection.txt"

echo "OURS_GPU_SELECTION status=PASS physical_gpus=$gpu_csv"
echo "OURS_CONTAINER name=$container_name image=$image image_id=$image_id scope=$scope"
if [[ "$ablation" == a2 ]]; then
  echo "OURS_DIAGNOSTIC_A2_CONTROL enabled=1 purpose=A3_numerical_isolation timing_eligible=0"
fi

diagnostic_env=()
timeout_seconds=1200
if [[ "$scope" == attention ]]; then
  diagnostic_env+=(OURS_VIT_STOP_AFTER_ATTENTION=1)
  timeout_seconds=900
elif [[ "$scope" == logits ]]; then
  diagnostic_env+=(OURS_VIT_PARTIAL_FINAL_LOGITS=1)
  timeout_seconds=1200
fi
if [[ "${OURS_VIT_CAPTURE_ATTENTION_TRACE:-0}" == 1 ]]; then
  if [[ "$scope" != attention || "$blocks" != 1 ]]; then
    echo "OURS_VIT_CAPTURE_ATTENTION_TRACE requires attention scope and BLOCKS=1" >&2
    exit 2
  fi
  diagnostic_env+=(
    OURS_VIT_HETERO_ATTENTION_TRACE=1
    OURS_VIT_HETERO_ATTENTION_TRACE_DIR="$output_dir"
  )
  echo "OURS_DIAGNOSTIC_TRACE enabled=1 feeds_secure_graph=0 timing_eligible=0"
fi

# Run the immutable ID captured before GPU selection.  The mutable
# ours:latest tag may be rebuilt by another experiment while this runner is
# preparing the host-side reorder oracle.
docker run --rm --pull=never --name "$container_name" \
  --network=none --cap-add=NET_ADMIN --ipc=host \
  --gpus "$gpu_request" \
  --mount "type=bind,src=$review_dir,dst=/workspace/OURS/outputs/a3_a4_review" \
  --mount "type=bind,src=$checkpoint_dir,dst=/workspace/OURS/checkpoints/vit_b16,readonly" \
  --mount "type=bind,src=$input_dir,dst=/workspace/OURS/inputs/vit_b16,readonly" \
  --mount "type=bind,src=$head_plan,dst=$container_plan,readonly" \
  --mount "type=bind,src=/usr/local/cuda-12.3,dst=/usr/local/cuda-12.3,readonly" \
  --workdir /workspace/OURS "$image_id" env \
  OURS_VIT_ABLATION="$ablation" \
  OURS_VIT_HEAD_PLAN="$container_plan" \
  OURS_VIT_PREVALIDATED_REORDER_BINDING="$output_dir/reorder_binding.txt" \
  OURS_VIT_NETWORK_MODE=netem \
  BLB_VIT_STOP_AFTER_BLOCK="$blocks" \
  BLB_VIT_TIMEOUT_SECONDS="$timeout_seconds" \
  "${diagnostic_env[@]}" \
  scripts/run_ours_vit_b16.sh gpu "$output_dir" "$port" 0 1

# The inner BLB runner writes its own GPU audit to the same filename. Append
# the immutable outer binding after a successful container exit so collectors
# can prove exactly which image and fixed plan produced the measurements.
{
  echo "outer_selection=dynamic_immediately_before_container_start"
  echo "image=$image"
  echo "image_id=$image_id"
  echo "plan=$head_plan"
  echo "plan_sha256=$(sha256sum "$head_plan" | awk '{print $1}')"
} >> "$host_output_dir/gpu_selection.txt"

if [[ "$scope" == full ]]; then
  python3 -B "$root/ours_vit_b16/scripts/summarize_a3_a4_run.py" \
    --run-dir "$host_output_dir"
fi

echo "OURS_A3_A4_CONTAINER_RUN status=PASS ablation=${ablation^^} scope=$scope output=$output_dir"
