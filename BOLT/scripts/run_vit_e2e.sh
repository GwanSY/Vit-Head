#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${BINARY:-$ROOT/SCI/build_vit/bin/BOLT_VIT}"
FIXTURE="${FIXTURE:-$ROOT/fixtures/vit_smoke}"
OUTPUT="${OUTPUT:-$ROOT/outputs/vit_smoke}"
PORT="${PORT:-47000}"
DEBUG_LOGITS="${DEBUG_LOGITS:-0}"
CHECK_EXPECTED="${CHECK_EXPECTED:-1}"

test -x "$BINARY"
test -f "$FIXTURE/alice_model.txt"
test -f "$FIXTURE/bob_image.txt"
mkdir -p "$OUTPUT"

alice_pid=""
cleanup() {
  if [[ -n "$alice_pid" ]] && kill -0 "$alice_pid" 2>/dev/null; then
    kill "$alice_pid" 2>/dev/null || true
    wait "$alice_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$BINARY" r=1 p="$PORT" model="$FIXTURE/alice_model.txt" \
  output="$OUTPUT/prediction.txt" trace_dir="$OUTPUT" debug_logits="$DEBUG_LOGITS" \
  >"$OUTPUT/alice.log" 2>&1 &
alice_pid=$!

"$BINARY" r=2 p="$PORT" ip=127.0.0.1 input="$FIXTURE/bob_image.txt" \
  output="$OUTPUT/prediction.txt" trace_dir="$OUTPUT" debug_logits="$DEBUG_LOGITS" \
  2>&1 | tee "$OUTPUT/bob.log"
wait "$alice_pid"
alice_pid=""

if [[ "$CHECK_EXPECTED" == 1 ]]; then
  diff -u "$FIXTURE/expected_prediction.txt" "$OUTPUT/prediction.txt"
  layers="$(awk '$1 == "layers" {print $2; exit}' "$FIXTURE/alice_model.txt")"
  if [[ "$layers" == 1 ]]; then
    echo "BOLT_VIT_SINGLE_LAYER_FINAL_CLASS=PASS" | tee -a "$OUTPUT/alice.log"
  else
    echo "BOLT_VIT_MULTI_LAYER_FINAL_CLASS=PASS" | tee -a "$OUTPUT/alice.log"
  fi
  test -s "$OUTPUT/backend_breakdown.csv"
  grep -q '^he,' "$OUTPUT/backend_breakdown.csv"
  grep -q '^mpc,' "$OUTPUT/backend_breakdown.csv"
  grep -q '^conversion,' "$OUTPUT/backend_breakdown.csv"
  touch "$OUTPUT/PASS"
  echo "PASS: $OUTPUT/prediction.txt"
else
  echo "COMPLETED_WITHOUT_REFERENCE_ASSERT: $OUTPUT/prediction.txt"
fi
