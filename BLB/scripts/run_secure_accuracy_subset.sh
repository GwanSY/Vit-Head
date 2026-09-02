#!/usr/bin/env bash
set -euo pipefail

# Run the full 12-block BLB ViT protocol on a fixed ImageNet subset.  The
# summary is append-only and successful rows are skipped, so a long accuracy
# run can be resumed safely.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-$WORKSPACE_ROOT/datasets/imagenet2012/val}"
MANIFEST="${1:-$WORKSPACE_ROOT/datasets/imagenet2012/splits/secure_random_seed20260830_n100.csv}"
OUTPUT_ROOT="${2:-$ROOT/outputs/secure_accuracy_seed20260830}"
BASE_PORT="${3:-64000}"
ALICE_GPU_ID="${4:-}"
BOB_GPU_ID="${5:-}"
CONTAINER_NAME="${BLB_CONTAINER_NAME:-blb}"
PYTHON="${PYTHON:-$WORKSPACE_ROOT/BOLT/.venv-vit-b16/bin/python}"

fail() {
  printf 'BLB secure accuracy: %s\n' "$*" >&2
  exit 2
}

[[ -s "$MANIFEST" ]] || fail "manifest not found or empty: $MANIFEST"
[[ -d "$DATASET_ROOT/images" ]] || fail "dataset images not found: $DATASET_ROOT/images"
[[ -x "$PYTHON" ]] || fail "Python interpreter not found: $PYTHON"
[[ -n "$ALICE_GPU_ID" && -n "$BOB_GPU_ID" ]] || \
  fail "usage: $0 MANIFEST OUTPUT_ROOT BASE_PORT ALICE_GPU_ID BOB_GPU_ID"
[[ "$BASE_PORT" =~ ^[0-9]+$ ]] || fail "BASE_PORT must be numeric"
case "$OUTPUT_ROOT" in
  "$ROOT"/*) ;;
  *) fail "OUTPUT_ROOT must be inside $ROOT so the container can access it" ;;
esac

mkdir -p "$OUTPUT_ROOT"
SUMMARY="$OUTPUT_ROOT/summary.csv"
if [[ ! -s "$SUMMARY" ]]; then
  printf '%s\n' \
    'index,image_path,true_label,float_class,fixed_class,secure_class,float_correct,fixed_correct,secure_correct,secure_matches_fixed,fixed_matches_float,status,return_code,reference_return_code,secure_online_s,setup_offline_s' \
    > "$SUMMARY"
fi

mapfile -t ROWS < "$MANIFEST"
total=0
for row in "${ROWS[@]}"; do
  [[ -n "${row//[[:space:]]/}" ]] || continue
  total=$((total + 1))
done
printf 'BLB_VIT_SECURE_SUBSET total=%s alice_gpu=%s bob_gpu=%s layers=12\n' \
  "$total" "$ALICE_GPU_ID" "$BOB_GPU_ID"

index=0
for row in "${ROWS[@]}"; do
  [[ -n "${row//[[:space:]]/}" ]] || continue
  index=$((index + 1))
  IFS=',' read -r relative_image true_label <<< "$row"
  sample_id="$(printf '%04d' "$index")"
  sample_output="$OUTPUT_ROOT/$sample_id"
  port=$((BASE_PORT + index - 1))

  if awk -F',' -v id="$index" \
      '$1 == id && $12 == "PASS" { found=1 } END { exit !found }' "$SUMMARY"; then
    printf 'BLB_VIT_SECURE_SAMPLE index=%s status=SKIP_ALREADY_PASS\n' "$index"
    continue
  fi

  image_path="$DATASET_ROOT/$relative_image"
  [[ -f "$image_path" ]] || fail "sample image not found: $image_path"
  mkdir -p "$sample_output"
  "$PYTHON" "$ROOT/vit_b16/scripts/preprocess_vit_image.py" "$image_path" \
    --manifest "$ROOT/checkpoints/vit_b16/manifest.json" \
    --output "$sample_output/bob_image.txt"

  relative_output="${sample_output#"$ROOT"/}"
  container_output="/workspace/BLB/$relative_output"
  container_input="$container_output/bob_image.txt"
  printf 'BLB_VIT_SECURE_SAMPLE index=%s/%s image=%s label=%s port=%s status=START\n' \
    "$index" "$total" "$relative_image" "$true_label" "$port"

  set +e
  docker exec "$CONTAINER_NAME" env \
    BLB_VIT_INPUT="$container_input" \
    BLB_VIT_TIMEOUT_SECONDS="${BLB_VIT_TIMEOUT_SECONDS:-21600}" \
    BLB_VIT_OMP_THREADS="${BLB_VIT_OMP_THREADS:-64}" \
    BLB_VIT_GPU_DIRECT_PLAINTEXT="${BLB_VIT_GPU_DIRECT_PLAINTEXT:-1}" \
    BLB_VIT_GPU_LINEAR_STREAMS="${BLB_VIT_GPU_LINEAR_STREAMS:-8}" \
    bash -lc 'cd /workspace/BLB && ./scripts/run_vit_b16.sh "$@"' \
    _ gpu "$relative_output" "$port" "$ALICE_GPU_ID" "$BOB_GPU_ID" \
    > "$sample_output/driver.log" 2>&1
  return_code=$?
  set -e

  pair_consistent=false
  if [[ -s "$sample_output/prediction_party1.txt" &&
        -s "$sample_output/prediction_party2.txt" &&
        -s "$sample_output/secure_logits_party1.txt" &&
        -s "$sample_output/secure_logits_party2.txt" ]] &&
      cmp -s "$sample_output/prediction_party1.txt" "$sample_output/prediction_party2.txt" &&
      cmp -s "$sample_output/secure_logits_party1.txt" "$sample_output/secure_logits_party2.txt"; then
    pair_consistent=true
  fi

  reference_return_code=1
  if [[ "$pair_consistent" == true ]]; then
    set +e
    "$PYTHON" "$ROOT/vit_b16/scripts/vit_reference.py" \
      --model "$ROOT/checkpoints/vit_b16/bolt_model.txt" \
      --input "$sample_output/bob_image.txt" \
      --secure-prediction "$sample_output/prediction_party1.txt" \
      --secure-logits "$sample_output/secure_logits_party1.txt" \
      --output "$sample_output/reference_report.json" \
      > "$sample_output/reference.log" 2>&1
    reference_return_code=$?
    set -e
  fi

  float_class=""
  fixed_class=""
  secure_class=""
  secure_matches_fixed=""
  fixed_matches_float=""
  if [[ -s "$sample_output/reference_report.json" ]]; then
    read -r float_class fixed_class secure_class secure_matches_fixed fixed_matches_float < <(
      "$PYTHON" - "$sample_output/reference_report.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    report.get("clear", {}).get("top1", ""),
    report.get("fixed", {}).get("top1", ""),
    report.get("secure", {}).get("top1", ""),
    report.get("secure", {}).get("top1_matches_fixed", ""),
    report.get("fixed", {}).get("top1_matches_clear", ""),
)
PY
    )
  fi

  float_correct=false
  fixed_correct=false
  secure_correct=false
  [[ -n "$float_class" && "$float_class" == "$true_label" ]] && float_correct=true
  [[ -n "$fixed_class" && "$fixed_class" == "$true_label" ]] && fixed_correct=true
  [[ -n "$secure_class" && "$secure_class" == "$true_label" ]] && secure_correct=true

  secure_online_s=""
  setup_offline_s=""
  if [[ -s "$sample_output/baseline_summary.csv" ]]; then
    secure_online_s="$(awk -F',' '$1 == "secure_end_to_end_latency_s" { print $2 }' \
      "$sample_output/baseline_summary.csv")"
    setup_offline_s="$(awk -F',' '$1 == "setup_offline_s" { print $2 }' \
      "$sample_output/baseline_summary.csv")"
  fi

  status=FAIL
  if [[ "$return_code" -eq 0 && "$pair_consistent" == true &&
        -n "$secure_class" && "$secure_matches_fixed" == True ]]; then
    status=PASS
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$index" "$relative_image" "$true_label" "$float_class" "$fixed_class" \
    "$secure_class" "$float_correct" "$fixed_correct" "$secure_correct" \
    "$secure_matches_fixed" "$fixed_matches_float" "$status" "$return_code" \
    "$reference_return_code" "$secure_online_s" "$setup_offline_s" >> "$SUMMARY"
  printf 'BLB_VIT_SECURE_SAMPLE index=%s secure=%s fixed=%s label=%s secure_correct=%s secure_matches_fixed=%s status=%s\n' \
    "$index" "$secure_class" "$fixed_class" "$true_label" "$secure_correct" \
    "$secure_matches_fixed" "$status"

  if [[ "$status" != PASS ]]; then
    printf 'BLB_VIT_SECURE_SUBSET_STOP index=%s reason=NUMERICAL_OR_PROTOCOL_GATE_FAILED\n' "$index" >&2
    exit 1
  fi
done

printf 'BLB_VIT_SECURE_SUBSET_DONE summary=%s\n' "$SUMMARY"
