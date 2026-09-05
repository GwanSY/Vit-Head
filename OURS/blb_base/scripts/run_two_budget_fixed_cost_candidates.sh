#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 TWO_BUDGET_CANDIDATE_MANIFEST RUN_TAG COST_OUTPUT_CSV" >&2
  exit 2
fi

manifest=$(readlink -f "$1")
tag=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cost_output=$3
if [[ "$cost_output" != /* ]]; then cost_output="$root/$cost_output"; fi
if [[ ! -f "$manifest" || ! "$tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid manifest or run tag" >&2
  exit 2
fi
if [[ "$cost_output" != "$root/outputs/"* ]]; then
  echo "cost output must be below outputs/" >&2
  exit 2
fi

run_root="outputs/a3_a4_review/two_budget_fixed_cost-$tag"
mkdir -p "$root/$run_root"
mkdir -p "$(dirname "$cost_output")"
fixed_pair="${OURS_VIT_PHYSICAL_GPUS:-1,7}"
IFS=, read -r fixed_gpu_a fixed_gpu_b <<< "$fixed_pair"
wait_for_pair() {
  while true; do
    uuid_a=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
      awk -F, -v i="$fixed_gpu_a" '$1 + 0 == i {gsub(/^ +| +$/, "", $2); print $2}')
    uuid_b=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
      awk -F, -v i="$fixed_gpu_b" '$1 + 0 == i {gsub(/^ +| +$/, "", $2); print $2}')
    ready=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free \
      --format=csv,noheader,nounits |
      awk -F, -v a="$fixed_gpu_a" -v b="$fixed_gpu_b" '
        ($1 + 0 == a || $1 + 0 == b) && $2 + 0 <= 5 && $3 + 0 >= 70000 {count++}
        END {print count + 0}')
    active=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
      --format=csv,noheader,nounits |
      awk -F, -v a="$uuid_a" -v b="$uuid_b" \
        '{gsub(/^ +| +$/, "", $1)} ($1 == a || $1 == b) && $3 + 0 > 10 {count++} END {print count + 0}')
    external=$(ps -eo pid=,comm=,args= | awk '
      $2 ~ /^python/ &&
      ($0 ~ /launch_paper_bandwidth.*\.py/ ||
       $0 ~ /run_paper_bandwidth\.py/) {count++}
      END {print count + 0}')
    [[ "$ready" == 2 && "$active" == 0 && "$external" == 0 ]] && return
    echo "OURS_TWO_BUDGET_FIXED_COST_WAIT pair=$fixed_pair status=WAIT"
    sleep 20
  done
}
mapfile -t candidate_rows < <(python3 -B -c \
  'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); [print(n,i["ablation"],i["path"],sep="\t") for n,i in p["plans"].items() if i.get("budget_group")=="fixed_candidate" and n!="Fixed-Sensitivity-only"]' \
  "$manifest")
if [[ ${#candidate_rows[@]} -eq 0 ]]; then
  echo "candidate manifest has no fixed candidates" >&2
  exit 2
fi

run_args=()
for candidate in "${candidate_rows[@]}"; do
  IFS=$'\t' read -r name ablation plan <<< "$candidate"
  slug=$(printf '%s' "$name" | tr '[:upper:] +/' '[:lower:]___')
  relative="$run_root/$slug"
  host="$root/$relative"
  run_args+=(--run "$host")
  if [[ -f "$host/status.txt" ]]; then
    if grep -q '^alice_rc=0$' "$host/status.txt" &&
       grep -q '^bob_rc=0$' "$host/status.txt"; then
      echo "OURS_TWO_BUDGET_FIXED_COST status=SKIP name=$name"
      continue
    fi
    mv "$host" "$host.failed-$(date +%s)"
    echo "OURS_TWO_BUDGET_FIXED_COST status=RETRY name=$name reason=prior_failure"
  fi
  if [[ -e "$host" ]]; then
    echo "incomplete fixed-cost run exists (preserved): $host" >&2
    exit 3
  fi
  echo "OURS_TWO_BUDGET_FIXED_COST status=START name=$name"
  while true; do
    wait_for_pair
    set +e
    OURS_DOCKER_IMAGE="${OURS_DOCKER_IMAGE:-ours:fast-preserved-20260901}" \
      OURS_VIT_PHYSICAL_GPUS="$fixed_pair" \
      BLB_VIT_TIMEOUT_SECONDS="${BLB_VIT_TIMEOUT_SECONDS:-1800}" \
      "$root/scripts/run_ours_a3_a4_container.sh" \
        "$ablation" "$plan" "$relative" 1 attention
    rc=$?
    set -e
    if [[ "$rc" == 0 ]]; then
      break
    fi
    # The launcher performs a second GPU check after the host-side reorder
    # oracle.  If an external job wins that small race, preserve the partial
    # provenance and retry the same candidate after the pair is idle.
    if [[ "$rc" == 3 && -d "$root/$relative" && ! -f "$root/$relative/status.txt" ]]; then
      mv "$root/$relative" "$root/${relative}.retry-${rc}-$(date +%s)"
      echo "OURS_TWO_BUDGET_FIXED_COST status=RETRY name=$name reason=gpu_race"
      continue
    fi
    exit "$rc"
  done
done

python3 -B "$root/ours_vit_b16/scripts/collect_attention_assignment_costs.py" \
  "${run_args[@]}" --output "$cost_output"
echo "OURS_TWO_BUDGET_FIXED_COST status=PASS output=$cost_output"
