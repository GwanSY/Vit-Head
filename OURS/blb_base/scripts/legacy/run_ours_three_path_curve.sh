#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_TAG" >&2
  exit 2
fi

tag=$1
if [[ ! "$tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
generator="$root/ours_vit_b16/scripts/legacy/make_three_path_profiler_plans.py"
collector="$root/ours_vit_b16/scripts/legacy/collect_three_path_costs.py"
plan_root="$root/outputs/paper/plans/three_path/$tag"
run_prefix="outputs/a3_a4_review/paper_three_path/$tag"
csv_root="$root/outputs/paper/three_path/$tag"
mkdir -p "$plan_root" "$csv_root"

# The curve is meaningful only when all 18 points use the same accelerators.
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
echo "OURS_THREE_PATH_GPU_BINDING status=PASS physical_gpus=$OURS_VIT_PHYSICAL_GPUS"

groups=(1 2 4 6 8 12)
paths=(softmax relu linear)
for path in "${paths[@]}"; do
  collector_args=()
  for group in "${groups[@]}"; do
    plan="$plan_root/${path}_g${group}.txt"
    run_rel="$run_prefix/${path}_g${group}"
    run_abs="$root/$run_rel"
    python3 -B "$generator" --path "$path" --group-size "$group" \
      --output "$plan"
    if [[ -f "$run_abs/status.txt" ]] &&
       grep -q '^alice_rc=0$' "$run_abs/status.txt" &&
       grep -q '^bob_rc=0$' "$run_abs/status.txt"; then
      echo "OURS_THREE_PATH_RUN status=SKIP_ALREADY_PASS path=$path group=$group"
    else
      if [[ -e "$run_abs" ]]; then
        echo "incomplete run exists; move it aside before retrying: $run_abs" >&2
        exit 3
      fi
      OURS_VIT_PORT=$((30001 + group * 20)) \
        "$root/scripts/run_ours_a3_a4_container.sh" \
        a4 "$plan" "$run_rel" 1 attention
    fi
    collector_args+=(--run "$run_abs")
  done
  case "$path" in
    softmax) formal_name=Softmax ;;
    relu) formal_name=ReLU ;;
    linear) formal_name=Linear ;;
    *) echo "unsupported path: $path" >&2; exit 2 ;;
  esac
  python3 -B "$collector" "${collector_args[@]}" \
    --include-path "$formal_name" \
    --output "$csv_root/${path}.csv"
done

python3 -B "$root/ours_vit_b16/scripts/legacy/merge_three_path_costs.py" \
  --input "$csv_root/softmax.csv" \
  --input "$csv_root/relu.csv" \
  --input "$csv_root/linear.csv" \
  --output "$csv_root/three_path_cost_curve.csv"

python3 -B "$root/ours_vit_b16/scripts/legacy/audit_paper_cohort.py" \
  --kind three-path \
  --expected-runs 18 \
  --run-root "$root/$run_prefix" \
  --output "$csv_root/cohort_audit.json"

echo "OURS_THREE_PATH_CURVE status=PASS csv=$csv_root/three_path_cost_curve.csv"
