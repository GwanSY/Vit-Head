#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 softmax|relu|linear|softmax11_relu1 OUTPUT_DIR" >&2
  exit 2
fi

path=$1
output_dir=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
image=${OURS_DOCKER_IMAGE:-ours}

case "$path" in
  softmax) plan=a2_all_softmax_head_plan.txt ;;
  relu) plan=a2_all_relu_head_plan.txt ;;
  linear) plan=a2_all_linear_head_plan.txt ;;
  softmax11_relu1) plan=a2_softmax11_relu1_head_plan.txt ;;
  *) echo "unknown A2 cost plan label: $path" >&2; exit 2 ;;
esac

if [[ "$image" != ours && "$image" != ours:* ]]; then
  echo "OURS_DOCKER_IMAGE must be ours or an ours:<tag> name" >&2
  exit 2
fi
if [[ "$output_dir" != outputs/a1_a2_review/* ]]; then
  echo "OUTPUT_DIR must be below outputs/a1_a2_review" >&2
  exit 2
fi
if [[ -e "$root/$output_dir" ]]; then
  echo "refusing to reuse existing output directory: $root/$output_dir" >&2
  exit 2
fi
if [[ ! -f "$root/outputs/a1_a2_review/$plan" ]]; then
  echo "missing homogeneous A2 plan: $plan" >&2
  exit 2
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "missing image $image; run scripts/build_ours_image.sh first" >&2
  exit 2
fi
image_id=$(docker image inspect "$image" --format '{{.Id}}')
if [[ -z "$image_id" ]]; then
  echo "failed to resolve immutable image ID for $image" >&2
  exit 2
fi

checkpoint_dir=$(readlink -f "$root/checkpoints/vit_b16")
input_dir=$(readlink -f "$root/inputs/vit_b16")
review_dir="$root/outputs/a1_a2_review"
host_output_dir="$root/$output_dir"
mkdir -p "$host_output_dir"

# The BLB base image has Python 3.6, while the independent oracle intentionally
# uses the host's audited Python/Numpy environment. Bind the successful result
# to the exact model, input, plan, block limit, and report hashes; the container
# verifies this file before opening either secure-party socket.
python3 -B "$root/ours_vit_b16/scripts/verify_head_reorder.py" \
  --model "$root/checkpoints/vit_b16/bolt_model.txt" \
  --input "$root/inputs/vit_b16/bob_image.txt" \
  --head-plan "$root/outputs/a1_a2_review/$plan" \
  --output "$host_output_dir/head_reorder_oracle.json" \
  --block-limit 1 > "$host_output_dir/head_reorder_oracle.log" 2>&1
binding="$host_output_dir/reorder_binding.txt"
{
  echo "format=OURS_VIT_REORDER_BINDING_V1"
  echo "status=PASS"
  echo "model_sha256=$(sha256sum "$checkpoint_dir/bolt_model.txt" | awk '{print $1}')"
  echo "input_sha256=$(sha256sum "$input_dir/bob_image.txt" | awk '{print $1}')"
  echo "plan_sha256=$(sha256sum "$root/outputs/a1_a2_review/$plan" | awk '{print $1}')"
  echo "block_limit=1"
  echo "oracle_report=$output_dir/head_reorder_oracle.json"
  echo "oracle_report_sha256=$(sha256sum "$host_output_dir/head_reorder_oracle.json" | awk '{print $1}')"
} > "$binding"

# A GPU is eligible only when utilization is small and at least 70 GiB remains
# free. Selection happens immediately before docker run and is never stored in
# the image or plan. Formal A2 measurements use the same two-party/two-GPU
# topology as the 498.104 s BLB baseline; sharing one physical GPU makes the
# normalization cost incomparable and is rejected by the collector.
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
container_name="ours-a2-${path}-$(date +%Y%m%d-%H%M%S)-$$"

{
  echo "selection=dynamic_immediately_before_container_start"
  echo "physical_gpus=$gpu_csv"
  echo "container_cuda_devices=0,1"
  echo "party_topology=separate_physical_gpus"
  echo "eligibility=utilization_le_5_percent_and_memory_free_ge_70000_MiB"
  echo "image=$image"
  echo "image_id=$image_id"
  date --iso-8601=ns
} > "$host_output_dir/gpu_selection.txt"

echo "OURS_GPU_SELECTION status=PASS physical_gpus=$gpu_csv"
echo "OURS_CONTAINER name=$container_name image=$image image_id=$image_id network=isolated party_topology=separate_physical_gpus"

docker run --rm --pull=never --name "$container_name" \
  --network=none --cap-add=NET_ADMIN --ipc=host \
  --gpus "$gpu_request" \
  --mount "type=bind,src=$review_dir,dst=/workspace/OURS/outputs/a1_a2_review" \
  --mount "type=bind,src=$checkpoint_dir,dst=/workspace/OURS/checkpoints/vit_b16,readonly" \
  --mount "type=bind,src=$input_dir,dst=/workspace/OURS/inputs/vit_b16,readonly" \
  --mount "type=bind,src=/usr/local/cuda-12.3,dst=/usr/local/cuda-12.3,readonly" \
  --workdir /workspace/OURS "$image" sh -lc \
  "env OURS_VIT_ABLATION=a2 \
OURS_VIT_HEAD_PLAN=outputs/a1_a2_review/$plan \
OURS_VIT_PREVALIDATED_REORDER_BINDING=$output_dir/reorder_binding.txt \
OURS_VIT_NETWORK_MODE=netem \
BLB_VIT_STOP_AFTER_BLOCK=1 \
OURS_VIT_STOP_AFTER_NORMALIZATION=1 \
BLB_VIT_TIMEOUT_SECONDS=1800 \
scripts/run_ours_vit_b16.sh gpu '$output_dir' 31001 0 1"
