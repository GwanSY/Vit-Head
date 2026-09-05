#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 VARIABLE_SUITE_CSV RUN_TAG WAIT_PAIR" >&2
  exit 2
fi

suite=$(readlink -f "$1")
tag=$2
pair=$3
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ ! -f "$suite" || ! "$tag" =~ ^[A-Za-z0-9._-]+$ ||
      ! "$pair" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "invalid suite, run tag, or GPU pair" >&2
  exit 2
fi
IFS=, read -r gpu_a gpu_b <<< "$pair"
if [[ "$gpu_a" == "$gpu_b" ]]; then
  echo "GPU pair must contain two distinct devices" >&2
  exit 2
fi

while true; do
  uuid_a=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
    awk -F, -v i="$gpu_a" '$1 + 0 == i {gsub(/^ +| +$/, "", $2); print $2}')
  uuid_b=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader |
    awk -F, -v i="$gpu_b" '$1 + 0 == i {gsub(/^ +| +$/, "", $2); print $2}')
  ready=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free \
    --format=csv,noheader,nounits |
    awk -F, -v a="$gpu_a" -v b="$gpu_b" '
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
  if [[ "$ready" == 2 && "$active" == 0 && "$external" == 0 ]]; then
    break
  fi
  echo "OURS_TWO_BUDGET_WAIT pair=$pair status=WAIT"
  sleep 20
done

echo "OURS_TWO_BUDGET_WAIT pair=$pair status=READY"
while true; do
  set +e
  OURS_DOCKER_IMAGE="${OURS_DOCKER_IMAGE:-ours:fast-preserved-20260901}" \
    OURS_VIT_PHYSICAL_GPUS="$pair" \
    BLB_VIT_TIMEOUT_SECONDS="${BLB_VIT_TIMEOUT_SECONDS:-21600}" \
    bash "$root/scripts/run_ours_full12_suite.sh" "$suite" "$tag" 1
  rc=$?
  set -e
  if [[ "$rc" == 0 ]]; then
    break
  fi
  if [[ "$rc" == 3 ]]; then
    # A new external job may start between the outer wait and the runner's
    # second GPU check.  Preserve only canonical run-XX directories.  Retry
    # directories intentionally have extra suffixes and are never selected.
    moved=0
    while IFS= read -r candidate; do
      candidate_name=${candidate##*/}
      if [[ "$candidate_name" =~ ^run-[0-9][0-9]$ &&
            ! -f "$candidate/status.txt" ]]; then
        retry_path="${candidate}.gpu-race-$(date +%s%N)"
        if mv "$candidate" "$retry_path" 2>/dev/null; then
          moved=1
          break
        fi
      fi
    done < <(find "$root/outputs/a3_a4_review/full12-$tag" \
      -mindepth 2 -maxdepth 2 -type d -print 2>/dev/null)
    if [[ "$moved" == 1 ]]; then
      echo "OURS_TWO_BUDGET status=RETRY reason=gpu_race"
    else
      # The suite-level GPU precheck can fail before it creates a run
      # directory.  Keep waiting for the same fixed cohort instead of
      # treating that transient race as a fatal experiment error.
      echo "OURS_TWO_BUDGET_WAIT pair=$pair status=RETRY_WAIT"
    fi
    sleep 20
    continue
  fi
  exit "$rc"
done
