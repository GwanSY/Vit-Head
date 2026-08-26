#!/usr/bin/env bash
set -euo pipefail

# Run the formal, full-depth BOLT ViT protocol on a fixed ImageNet subset.
# This is an experiment harness only: it does not alter the secure operator
# graph or enable a layer-prefix validation mode.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:-$ROOT/datasets/imagenet2012/val/validation_manifest_100.csv}"
OUTPUT_ROOT="${2:-$ROOT/outputs/secure_accuracy}"
BASE_PORT="${3:-49201}"

: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must name the physical GPU for this run}"
BINARY="${BINARY:-$ROOT/SCI/build_gpu/bin/BOLT_VIT_GPU}"
PYTHON="${PYTHON:-$ROOT/.venv-vit-b16/bin/python}"
BOLT_GPU_RESIDENT="${BOLT_GPU_RESIDENT:-0}"

[[ -s "$MANIFEST" ]] || { echo "manifest not found or empty: $MANIFEST" >&2; exit 2; }
[[ -x "$BINARY" ]] || { echo "executable not found: $BINARY" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Python interpreter not found: $PYTHON" >&2; exit 2; }

mkdir -p "$OUTPUT_ROOT"
SUMMARY="$OUTPUT_ROOT/summary.csv"
if [[ ! -s "$SUMMARY" ]]; then
  printf 'index,image_path,true_label,predicted_class,fixed_class,secure_matches_fixed,secure_correct,status,return_code,communication_bytes,end_to_end_time_ms\n' > "$SUMMARY"
fi

mapfile -t ROWS < "$MANIFEST"
total=0
for row in "${ROWS[@]}"; do
  [[ -n "${row//[[:space:]]/}" ]] || continue
  total=$((total + 1))
done
echo "BOLT_VIT_SECURE_SUBSET total=$total gpu=$CUDA_VISIBLE_DEVICES layers=12"

index=0
for row in "${ROWS[@]}"; do
  [[ -n "${row//[[:space:]]/}" ]] || continue
  index=$((index + 1))
  IFS=',' read -r relative_image true_label <<< "$row"
  sample_id="$(printf '%04d' "$index")"
  sample_output="$OUTPUT_ROOT/$sample_id"
  port=$((BASE_PORT + index - 1))

  # A PASS row is durable evidence that this item completed.  This makes an
  # interrupted batch resumable without rerunning completed secure sessions.
  if awk -F',' -v id="$index" '$1 == id && $8 == "PASS" { found=1 } END { exit !found }' "$SUMMARY"; then
    echo "BOLT_VIT_SECURE_SAMPLE index=$index status=SKIP_ALREADY_PASS"
    continue
  fi

  image_path="$ROOT/datasets/imagenet2012/val/$relative_image"
  mkdir -p "$sample_output"
  echo "BOLT_VIT_SECURE_SAMPLE index=$index/$total image=$relative_image label=$true_label port=$port status=START"

  set +e
  env -u BOLT_VIT_MAX_LAYERS \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    BOLT_GPU_RESIDENT="$BOLT_GPU_RESIDENT" \
    BINARY="$BINARY" PYTHON="$PYTHON" \
    "$ROOT/scripts/run_vit_gpu.sh" "$image_path" "$sample_output" "$port" \
    > "$sample_output/driver.log" 2>&1
  return_code=$?
  set -e

  predicted_class=""
  if [[ -s "$sample_output/prediction.txt" ]]; then
    predicted_class="$(sed -n 's/.*predicted_class[ =]\([0-9][0-9]*\).*/\1/p; s/.*class[= ]\([0-9][0-9]*\).*/\1/p' "$sample_output/prediction.txt" | tail -1)"
  fi
  fixed_class=""
  secure_matches_fixed=""
  if [[ -s "$sample_output/reference_report.json" ]]; then
    read -r fixed_class secure_matches_fixed < <(
      "$PYTHON" - "$sample_output/reference_report.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
fixed = report.get("fixed", {})
secure = report.get("secure", {})
print(fixed.get("top1", ""), secure.get("top1_matches_fixed", ""))
PY
    )
  fi
  secure_correct="false"
  if [[ -n "$predicted_class" && "$predicted_class" == "$true_label" ]]; then
    secure_correct="true"
  fi
  status="FAIL"
  if [[ "$return_code" -eq 0 && -n "$predicted_class" && -n "$fixed_class" && "$secure_matches_fixed" == "True" ]]; then
    status="PASS"
  fi

  communication_bytes=""
  end_to_end_time_ms=""
  if [[ -s "$sample_output/alice.log" ]]; then
    metrics_line="$(rg '^BOLT_VIT_E2E_METRICS ' "$sample_output/alice.log" "$sample_output/bob.log" 2>/dev/null | tail -1 || true)"
    communication_bytes="$(sed -n 's/.*communication_bytes=\([0-9][0-9]*\).*/\1/p' <<< "$metrics_line")"
    end_to_end_time_ms="$(sed -n 's/.*end_to_end_time_ms=\([0-9.][0-9.]*\).*/\1/p' <<< "$metrics_line")"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$index" "$relative_image" "$true_label" "$predicted_class" "$fixed_class" \
    "$secure_matches_fixed" "$secure_correct" "$status" "$return_code" \
    "$communication_bytes" "$end_to_end_time_ms" >> "$SUMMARY"
  echo "BOLT_VIT_SECURE_SAMPLE index=$index predicted=$predicted_class fixed=$fixed_class secure_correct=$secure_correct secure_matches_fixed=$secure_matches_fixed status=$status return_code=$return_code"
done

echo "BOLT_VIT_SECURE_SUBSET_DONE summary=$SUMMARY"
