#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 SUITE_CSV RUN_TAG [REPEATS]" >&2
  exit 2
fi

suite=$(readlink -f "$1")
tag=$2
repeats=${3:-3}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [[ ! -f "$suite" || ! "$tag" =~ ^[A-Za-z0-9._-]+$ ||
      ! "$repeats" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid suite, tag, or repeat count" >&2
  exit 2
fi

if [[ -n "${OURS_VIT_PHYSICAL_GPUS:-}" ]]; then
  if [[ ! "$OURS_VIT_PHYSICAL_GPUS" =~ ^[0-9]+,[0-9]+$ ]]; then
    echo "OURS_VIT_PHYSICAL_GPUS must be two comma-separated GPU indices" >&2
    exit 2
  fi
  IFS=, read -r -a selected_gpus <<< "$OURS_VIT_PHYSICAL_GPUS"
  if [[ "${selected_gpus[0]}" == "${selected_gpus[1]}" ]]; then
    echo "OURS_VIT_PHYSICAL_GPUS must contain two distinct GPUs" >&2
    exit 2
  fi
  mapfile -t eligible_gpus < <(
    nvidia-smi --query-gpu=index,utilization.gpu,memory.free \
      --format=csv,noheader,nounits |
    awk -F, '$2 + 0 <= 5 && $3 + 0 >= 70000 {print $1 + 0}'
  )
  for selected_gpu in "${selected_gpus[@]}"; do
    found=0
    for eligible_gpu in "${eligible_gpus[@]}"; do
      [[ "$selected_gpu" == "$eligible_gpu" ]] && found=1
    done
    if [[ "$found" != 1 ]]; then
      echo "pre-bound GPU $selected_gpu is not idle with at least 70000 MiB free" >&2
      exit 3
    fi
  done
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
  if [[ ${#selected_gpus[@]} -ne 2 ||
        "${selected_gpus[0]}" == "${selected_gpus[1]}" ]]; then
    echo "need two distinct idle GPUs with at least 70000 MiB free" >&2
    exit 3
  fi
  export OURS_VIT_PHYSICAL_GPUS=${selected_gpus[0]},${selected_gpus[1]}
fi

base="outputs/a3_a4_review/full12-$tag"
host_base="$root/$base"
mkdir -p "$host_base"
while IFS=$'\t' read -r name ablation packing plan; do
  slug=$(printf '%s' "$name" | tr '[:upper:] +/' '[:lower:]___')
  for ((repeat=1; repeat<=repeats; ++repeat)); do
    suffix=$(printf '%02d' "$repeat")
    relative="$base/$slug/run-$suffix"
    host="$root/$relative"
    if [[ -f "$host/status.txt" ]]; then
      if grep -q '^alice_rc=0$' "$host/status.txt" &&
         grep -q '^bob_rc=0$' "$host/status.txt"; then
        echo "OURS_FULL12_RESUME status=SKIP name=$name repeat=$repeat"
        continue
      fi
      mv "$host" "$host.failed-$(date +%s)"
      echo "OURS_FULL12_RESUME status=RETRY name=$name repeat=$repeat reason=prior_failure"
    fi
    if [[ -e "$host" ]]; then
      echo "incomplete run already exists (preserved): $host" >&2
      exit 4
    fi
    extra_env=()
    [[ "$ablation" == a0 ]] && extra_env+=(OURS_VIT_FORMAL_A0_BASELINE=1)
    [[ "$ablation" == a2 ]] && extra_env+=(OURS_VIT_FORMAL_A2_SELECTED=1)
    env "${extra_env[@]}" OURS_VIT_GROUP_PACKING="$packing" \
      "$root/scripts/run_ours_a3_a4_container.sh" \
        "$ablation" "$plan" "$relative" 12 full
  done
done < <(python3 -B - "$suite" <<'PY'
import csv
import sys
for row in csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8")):
    if row["ablation"].lower() not in {"a0", "a2", "a3", "a4"}:
        raise ValueError("unsupported ablation")
    packing = row["packing"] or "1"
    if packing not in {"0", "1"}:
        raise ValueError("packing must be 0 or 1")
    print(row["name"], row["ablation"].lower(), packing,
          row["head_plan"], sep="\t")
PY
)

variant_count=$(python3 -B - "$suite" <<'PY'
import csv, sys
print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"))))
PY
)
python3 -B "$root/ours_vit_b16/scripts/legacy/audit_paper_cohort.py" \
  --kind full12 --expected-runs "$((variant_count * repeats))" \
  --run-root "$host_base" --output "$host_base/cohort_audit.json"
python3 -B "$root/ours_vit_b16/scripts/legacy/summarize_full12_suite.py" \
  --suite "$suite" --run-root "$host_base" \
  --output "$host_base/full12_suite.json"

echo "OURS_FULL12_SUITE status=PASS output=$base"
