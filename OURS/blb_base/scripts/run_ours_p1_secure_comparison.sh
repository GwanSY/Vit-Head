#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 PLAN_MANIFEST IMAGE_MANIFEST RUN_TAG [LIMIT]" >&2
  exit 2
fi

plan_manifest=$(readlink -f "$1")
image_manifest=$(readlink -f "$2")
tag=$3
limit=${4:-50}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ ! -f "$plan_manifest" || ! -f "$image_manifest" ]]; then
  echo "missing plan or image manifest" >&2
  exit 2
fi
if [[ ! "$tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG contains unsupported characters" >&2
  exit 2
fi
if [[ "$limit" -lt 50 ]]; then
  echo "formal P1 comparison requires at least 50 images" >&2
  exit 2
fi

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
  echo "need two distinct idle GPUs with at least 70000 MiB free" >&2
  exit 3
fi
export OURS_VIT_PHYSICAL_GPUS=${selected_gpus[0]},${selected_gpus[1]}
echo "OURS_P1_GPU_BINDING status=PASS physical_gpus=$OURS_VIT_PHYSICAL_GPUS"

while IFS=$'\t' read -r name plan linear_count; do
  run_tag="$tag-$(printf '%s' "$name" | tr '[:upper:] -' '[:lower:]__')"
  ablation=a3
  ((linear_count > 0)) && ablation=a4
  echo "OURS_P1_METHOD status=START name=$name ablation=$ablation tag=$run_tag"
  "$root/scripts/run_ours_secure_accuracy.sh" \
    "$ablation" "$plan" "$image_manifest" "$run_tag" "$limit"
done < <(python3 -B - "$plan_manifest" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for name, info in manifest["plans"].items():
    print(name, info["path"], info["global_counts"].get("he_linear", 0), sep="\t")
PY
)

run_root="$root/outputs/a3_a4_review/paper_secure_accuracy"
output_dir="$root/outputs/paper/p1-$tag"
mkdir -p "$output_dir"
python3 -B "$root/ours_vit_b16/scripts/summarize_p1_secure.py" \
  --plan-manifest "$plan_manifest" --run-root "$run_root" \
  --tag-prefix "$tag" --output "$output_dir/p1_secure_comparison.json"

echo "OURS_P1_SECURE_COMPARISON status=PASS output=$output_dir"
