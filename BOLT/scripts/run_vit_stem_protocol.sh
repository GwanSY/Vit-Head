#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORMAL="${FORMAL_BINARY:-$ROOT/SCI/build_vit/bin/BOLT_VIT_STEM_PROTOCOL_TEST}"
VALIDATION="${VALIDATION_BINARY:-$ROOT/SCI/build_vit/bin/BOLT_VIT_STEM_PROTOCOL_VALIDATION}"
MODEL="${MODEL:-$ROOT/fixtures/vit_smoke/alice_model.txt}"
IMAGE="${IMAGE:-$ROOT/fixtures/vit_smoke/bob_image.txt}"
OUTPUT="${OUTPUT:-$ROOT/outputs/vit_stem_protocol}"
PORT="${PORT:-47400}"

test -x "$FORMAL"
test -x "$VALIDATION"
mkdir -p "$OUTPUT"
validation_tmp="$(mktemp -d)"
alice_pid=""
cleanup() {
  if [[ -n "$alice_pid" ]] && kill -0 "$alice_pid" 2>/dev/null; then
    kill "$alice_pid" 2>/dev/null || true
    wait "$alice_pid" 2>/dev/null || true
  fi
  rm -rf "$validation_tmp"
}
trap cleanup EXIT INT TERM

"$VALIDATION" r=1 p="$PORT" model="$MODEL" image="$IMAGE" out="$OUTPUT" \
  >"$validation_tmp/alice.log" 2>&1 &
alice_pid=$!
"$VALIDATION" r=2 p="$PORT" ip=127.0.0.1 model="$MODEL" image="$IMAGE" \
  out="$OUTPUT" >"$validation_tmp/bob.log" 2>&1
wait "$alice_pid"
alice_pid=""
test -s "$OUTPUT/numerical_match.csv"
! grep -q ',FAIL$' "$OUTPUT/numerical_match.csv"
test "$(wc -l < "$OUTPUT/numerical_match.csv")" -eq 98

formal_port=$((PORT + 1))
"$FORMAL" r=1 p="$formal_port" model="$MODEL" \
  image="$validation_tmp/alice_must_not_read_image" out="$OUTPUT" \
  validation="$OUTPUT/numerical_match.csv" >"$OUTPUT/alice.log" 2>&1 &
alice_pid=$!
"$FORMAL" r=2 p="$formal_port" ip=127.0.0.1 \
  model="$validation_tmp/bob_must_not_read_model" image="$IMAGE" \
  out="$OUTPUT" validation="$OUTPUT/numerical_match.csv" >"$OUTPUT/bob.log" 2>&1
wait "$alice_pid"
alice_pid=""

test -s "$OUTPUT/operator_trace.csv"
test -s "$OUTPUT/packing_metadata.json"
test -s "$OUTPUT/first_mismatch.json"
grep -q '^VIT_STEM_NUMERICAL_MATCH status=PASS$' "$OUTPUT/alice.log"
grep -q '^PLAINTEXT_INTERMEDIATE_RECONSTRUCTIONS=0$' "$OUTPUT/alice.log"
grep -q '^SCALAR_FALLBACKS=0$' "$OUTPUT/alice.log"
grep -q '^GENERIC_OT_MATMUL_CALLS=0$' "$OUTPUT/alice.log"
grep -q '^M3A_VIT_STEM_LN1_PROTOCOL=PASS$' "$OUTPUT/alice.log"
cat "$OUTPUT/alice.log"
