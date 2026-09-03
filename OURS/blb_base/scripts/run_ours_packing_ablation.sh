#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 a3|a4 HEAD_PLAN RUN_TAG [REPEATS]" >&2
  exit 2
fi

ablation=${1,,}
head_plan=$2
tag=$3
repeats=${4:-3}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ "$ablation" != a3 && "$ablation" != a4 ]]; then
  echo "ablation must be a3 or a4" >&2
  exit 2
fi
if [[ ! "$tag" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG may contain only letters, digits, dot, underscore and dash" >&2
  exit 2
fi
if [[ ! "$repeats" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPEATS must be a positive integer" >&2
  exit 2
fi
if [[ "$head_plan" != /* ]]; then head_plan="$root/$head_plan"; fi
head_plan=$(readlink -f "$head_plan")
if [[ ! -f "$head_plan" ]]; then
  echo "missing head plan: $head_plan" >&2
  exit 2
fi

base="outputs/a3_a4_review/packing-$tag"
host_base="$root/$base"
mkdir -p "$host_base/timing/packed" "$host_base/timing/unpacked" \
         "$host_base/trace"

# Bind both variants to the same two physical accelerators for the complete
# paired cohort. Each child run records full UUID/name/driver telemetry.
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
  echo "need two idle GPUs with at least 70000 MiB free" >&2
  exit 3
fi
export OURS_VIT_PHYSICAL_GPUS=${selected_gpus[0]},${selected_gpus[1]}
echo "OURS_PACKING_GPU_BINDING status=PASS physical_gpus=$OURS_VIT_PHYSICAL_GPUS"

run_one() {
  local variant=$1
  local packing=$2
  local relative=$3
  local host="$root/$relative"
  if [[ -f "$host/ours_summary.csv" ]]; then
    echo "OURS_PACKING_RESUME status=SKIP variant=$variant output=$relative"
    return
  fi
  if [[ -e "$host" ]]; then
    echo "incomplete run already exists (preserved): $host" >&2
    exit 4
  fi
  OURS_VIT_GROUP_PACKING=$packing \
    "$root/scripts/run_ours_a3_a4_container.sh" \
      "$ablation" "$head_plan" "$relative" 12 full
}

for ((repeat=1; repeat<=repeats; ++repeat)); do
  suffix=$(printf '%02d' "$repeat")
  # Alternate the execution order to reduce systematic warm/cold bias.
  if ((repeat % 2 == 1)); then
    run_one packed 1 "$base/timing/packed/run-$suffix"
    run_one unpacked 0 "$base/timing/unpacked/run-$suffix"
  else
    run_one unpacked 0 "$base/timing/unpacked/run-$suffix"
    run_one packed 1 "$base/timing/packed/run-$suffix"
  fi
done

python3 -B "$root/ours_vit_b16/scripts/audit_paper_cohort.py" \
  --kind packing --expected-runs "$((2 * repeats))" \
  --run-root "$host_base/timing" \
  --output "$host_base/timing/cohort_audit.json"
python3 -B "$root/ours_vit_b16/scripts/summarize_packing_ablation.py" \
  --run-root "$host_base/timing" \
  --output "$host_base/packing_ablation_summary.json"

# Trace runs are separate and explicitly timing-ineligible. They run a real
# classified block-1 prefix so QK/Norm/AV/W_O/residual/logits are all compared.
for variant in packed unpacked; do
  packing=1
  [[ "$variant" == unpacked ]] && packing=0
  relative="$base/trace/$variant"
  host="$root/$relative"
  if [[ ! -f "$host/status.txt" ]]; then
    if [[ -e "$host" ]]; then
      echo "incomplete trace already exists (preserved): $host" >&2
      exit 4
    fi
    OURS_VIT_GROUP_PACKING=$packing \
    OURS_VIT_CAPTURE_ATTENTION_TRACE=1 \
      "$root/scripts/run_ours_a3_a4_container.sh" \
        "$ablation" "$head_plan" "$relative" 1 logits
  fi
done
python3 -B "$root/ours_vit_b16/scripts/audit_paper_cohort.py" \
  --kind packing --expected-runs 2 \
  --run-root "$host_base/trace" \
  --output "$host_base/trace/cohort_audit.json"
python3 -B "$root/ours_vit_b16/scripts/compare_packing_traces.py" \
  --packed "$host_base/trace/packed" \
  --unpacked "$host_base/trace/unpacked" \
  --head-plan "$head_plan" \
  --output "$host_base/packing_trace_comparison.json"

echo "OURS_PACKING_ABLATION status=PASS output=$base"
