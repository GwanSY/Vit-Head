#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 a0|a2|a3|a4 HEAD_PLAN MANIFEST RUN_TAG [LIMIT]" >&2
  exit 2
fi

ablation=${1,,}
head_plan=$(readlink -f "$2")
manifest=$(readlink -f "$3")
tag=$4
limit=${5:-50}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
workspace_root=$(cd "$root/../.." && pwd -P)
dataset_root=${DATASET_ROOT:-$workspace_root/datasets/imagenet2012/val}
python_bin=${PYTHON:-python3}
base_port=${OURS_VIT_BASE_PORT:-25001}
docker_image=${OURS_DOCKER_IMAGE:-ours}
secure_timeout_seconds=${OURS_VIT_SECURE_TIMEOUT_SECONDS:-21600}

if [[ "$ablation" != a0 && "$ablation" != a2 &&
      "$ablation" != a3 && "$ablation" != a4 ]]; then
  echo "ablation must be a0, a2, a3, or a4" >&2
  exit 2
fi
if [[ ! "$tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi
if [[ ! "$limit" =~ ^[1-9][0-9]*$ || "$limit" -gt 1000 ]]; then
  echo "LIMIT must be an integer in [1, 1000]" >&2
  exit 2
fi
if [[ ! "$secure_timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "OURS_VIT_SECURE_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
for path in "$head_plan" "$manifest" \
            "$root/checkpoints/vit_b16/manifest.json" \
            "$root/checkpoints/vit_b16/bolt_model.txt"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 2
  fi
done
if [[ "$docker_image" != ours && "$docker_image" != ours:* ]]; then
  echo "OURS_DOCKER_IMAGE must be ours or an ours:<tag> name" >&2
  exit 2
fi
if ! docker image inspect "$docker_image" >/dev/null 2>&1; then
  echo "missing image $docker_image" >&2
  exit 2
fi
cohort_image_id=$(docker image inspect "$docker_image" --format '{{.Id}}')

# A secure accuracy cohort is one experiment, not 50 unrelated placements.
# Bind the physical pair once unless an outer fair-comparison runner already
# supplied it, then let every per-image run audit the same pair.
if [[ -z "${OURS_VIT_PHYSICAL_GPUS:-}" ]]; then
  mapfile -t selected_gpus < <(
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free \
      --format=csv,noheader,nounits |
    awk -F, '$2 + 0 <= 5 && $4 + 0 >= 70000 {
                print $1 + 0, $3 + 0, $4 + 0
              }' |
    sort -k2,2n -k3,3nr |
    awk 'NR <= 2 {print $1}'
  )
  if [[ ${#selected_gpus[@]} -ne 2 ]]; then
    echo "need two idle GPUs with at least 70000 MiB free" >&2
    exit 3
  fi
  export OURS_VIT_PHYSICAL_GPUS=${selected_gpus[0]},${selected_gpus[1]}
fi
if [[ ! "$OURS_VIT_PHYSICAL_GPUS" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "OURS_VIT_PHYSICAL_GPUS must be two comma-separated GPU indices" >&2
  exit 2
fi
echo "OURS_SECURE_ACCURACY_COHORT_BINDING status=PASS physical_gpus=$OURS_VIT_PHYSICAL_GPUS image_id=$cohort_image_id"

manifest_sha=$(sha256sum "$manifest" | awk '{print $1}')
input_cache="$root/outputs/paper/preprocessed/$manifest_sha"
run_rel_root="outputs/a3_a4_review/paper_secure_accuracy/$tag"
run_abs_root="$root/$run_rel_root"
mkdir -p "$input_cache" "$run_abs_root"

mapfile -t rows < "$manifest"
sample_index=0
for row in "${rows[@]}"; do
  [[ -n "${row//[[:space:]]/}" ]] || continue
  sample_index=$((sample_index + 1))
  if [[ $sample_index -gt $limit ]]; then break; fi
  IFS=',' read -r relative_image true_label extra <<< "$row"
  if [[ -z "$relative_image" || -z "$true_label" || -n "${extra:-}" ||
        ! "$true_label" =~ ^[0-9]+$ ]]; then
    echo "manifest row $sample_index must be image_path,zero_based_label" >&2
    exit 2
  fi
  sample_image="$dataset_root/$relative_image"
  if [[ ! -f "$sample_image" ]]; then
    echo "missing ImageNet sample: $sample_image" >&2
    exit 2
  fi
  sample_id=$(printf '%04d' "$sample_index")
  input="$input_cache/$sample_id.txt"
  run_rel="$run_rel_root/$sample_id"
  run_abs="$root/$run_rel"
  if [[ ! -s "$input" ]]; then
    "$python_bin" -B "$root/ours_vit_b16/scripts/preprocess_vit_image.py" \
      "$sample_image" --manifest "$root/checkpoints/vit_b16/manifest.json" \
      --output "$input"
  fi
  if [[ -s "$run_abs/sample_result.json" ]]; then
    echo "OURS_SECURE_ACCURACY_SAMPLE index=$sample_index status=SKIP_ALREADY_RECORDED"
    continue
  fi
  if [[ -e "$run_abs" ]]; then
    echo "incomplete sample run exists; move it aside before retrying: $run_abs" >&2
    exit 3
  fi
  port=$((base_port + sample_index - 1))
  current_image_id=$(docker image inspect "$docker_image" --format '{{.Id}}')
  if [[ "$current_image_id" != "$cohort_image_id" ]]; then
    echo "image tag changed during secure cohort: expected=$cohort_image_id actual=$current_image_id" >&2
    exit 4
  fi
  echo "OURS_SECURE_ACCURACY_SAMPLE index=$sample_index/$limit image=$relative_image label=$true_label status=START"
  formal_env=()
  [[ "$ablation" == a0 ]] && formal_env+=(OURS_VIT_FORMAL_A0_BASELINE=1)
  [[ "$ablation" == a2 ]] && formal_env+=(OURS_VIT_FORMAL_A2_SELECTED=1)
  OURS_VIT_INPUT="$input" OURS_VIT_PORT="$port" \
    BLB_VIT_TIMEOUT_SECONDS="$secure_timeout_seconds" \
    env "${formal_env[@]}" "$root/scripts/run_ours_a3_a4_container.sh" \
    "$ablation" "$head_plan" "$run_rel" 12 full

  "$python_bin" -B "$root/ours_vit_b16/scripts/vit_reference.py" \
    --model "$root/checkpoints/vit_b16/bolt_model.txt" \
    --input "$input" \
    --head-plan "$head_plan" \
    --secure-prediction "$run_abs/prediction_party1.txt" \
    --secure-logits "$run_abs/secure_logits_party1.txt" \
    --output "$run_abs/reference_report.json" \
    > "$run_abs/reference.log" 2>&1

  "$python_bin" -B "$root/ours_vit_b16/scripts/legacy/record_secure_accuracy_sample.py" \
    --run-dir "$run_abs" \
    --reference "$run_abs/reference_report.json" \
    --index "$sample_index" \
    --image "$relative_image" \
    --label "$true_label" \
    --manifest "$manifest" \
    --head-plan "$head_plan" \
    --output "$run_abs/sample_result.json"
done

if [[ $sample_index -lt $limit ]]; then
  echo "manifest contains only $sample_index non-empty rows; requested $limit" >&2
  exit 2
fi

"$python_bin" -B "$root/ours_vit_b16/scripts/legacy/summarize_secure_accuracy.py" \
  --run-root "$run_abs_root" \
  --manifest "$manifest" \
  --head-plan "$head_plan" \
  --expected-samples "$limit" \
  --output "$run_abs_root/secure_accuracy.json"

echo "OURS_SECURE_ACCURACY status=PASS samples=$limit output=$run_abs_root/secure_accuracy.json"
