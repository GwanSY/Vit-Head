#!/usr/bin/env bash
set -euo pipefail

IRON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$IRON_ROOT/.." && pwd)"
# Preserve the existing tiny-fixture regression semantics.  The standard
# 224/16 runner uses IRON_VIT (the compact B/16 adapter) explicitly.
BINARY="${BINARY:-$IRON_ROOT/build/secure_vit/bin/IRON_VIT_GENERIC}"
FIXTURE="${FIXTURE:-$IRON_ROOT/fixtures/vit_smoke}"
OUTPUT="${OUTPUT:-$IRON_ROOT/outputs/vit_smoke}"
PORT="${PORT:-47100}"
DEBUG_LOGITS="${DEBUG_LOGITS:-0}"
RUN_TIMEOUT="${RUN_TIMEOUT:-10m}"

test -x "$BINARY"
if [[ ! -f "$FIXTURE/alice_model.txt" || ! -f "$FIXTURE/bob_image.txt" ]]; then
  python3 "$WORKSPACE_ROOT/BOLT/scripts/generate_vit_fixture.py" \
    --protocol IRON --out-dir "$FIXTURE"
fi
test -f "$FIXTURE/expected_prediction.txt"
mkdir -p "$OUTPUT"
rm -f "$OUTPUT/PASS"

alice_pid=""
cleanup() {
  if [[ -n "$alice_pid" ]] && kill -0 "$alice_pid" 2>/dev/null; then
    kill "$alice_pid" 2>/dev/null || true
    wait "$alice_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

timeout "$RUN_TIMEOUT" "$BINARY" r=1 p="$PORT" \
  model="$FIXTURE/alice_model.txt" \
  output="$OUTPUT/prediction.txt" trace_dir="$OUTPUT" \
  debug_logits="$DEBUG_LOGITS" \
  >"$OUTPUT/alice.log" 2>&1 &
alice_pid=$!

timeout "$RUN_TIMEOUT" "$BINARY" r=2 p="$PORT" ip=127.0.0.1 \
  input="$FIXTURE/bob_image.txt" \
  output="$OUTPUT/prediction.txt" trace_dir="$OUTPUT" \
  debug_logits="$DEBUG_LOGITS" \
  2>&1 | tee "$OUTPUT/bob.log"
wait "$alice_pid"
alice_pid=""

diff -u "$FIXTURE/expected_prediction.txt" "$OUTPUT/prediction.txt"
printf 'IRON ViT end-to-end secure inference passed.\n' >"$OUTPUT/PASS"
echo "PASS: $OUTPUT/prediction.txt"
