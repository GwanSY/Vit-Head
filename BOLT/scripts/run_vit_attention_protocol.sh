#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${BINARY:-$ROOT/SCI/build_vit/bin/BOLT_VIT_ATTENTION_PROTOCOL_TEST}"
OUTPUT="${OUTPUT:-$ROOT/outputs/vit_attention_protocol}"
PORT="${PORT:-47300}"

test -x "$BINARY"
mkdir -p "$OUTPUT"

alice_pid=""
cleanup() {
  if [[ -n "$alice_pid" ]] && kill -0 "$alice_pid" 2>/dev/null; then
    kill "$alice_pid" 2>/dev/null || true
    wait "$alice_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$BINARY" r=1 p="$PORT" out="$OUTPUT" >"$OUTPUT/alice.log" 2>&1 &
alice_pid=$!
"$BINARY" r=2 p="$PORT" ip=127.0.0.1 out="$OUTPUT" >"$OUTPUT/bob.log" 2>&1
wait "$alice_pid"
alice_pid=""

test -s "$OUTPUT/operator_trace.csv"
test -s "$OUTPUT/numerical_match.csv"
test -s "$OUTPUT/packing_metadata.json"
test -s "$OUTPUT/first_mismatch.json"
grep -q '^VIT_ATTENTION_OP op=qkv$' "$OUTPUT/alice.log"
grep -q '^VIT_ATTENTION_OP op=qk$' "$OUTPUT/alice.log"
grep -q '^VIT_ATTENTION_OP op=softmax$' "$OUTPUT/alice.log"
grep -q '^VIT_ATTENTION_OP op=softmax_v$' "$OUTPUT/alice.log"
grep -q '^VIT_ATTENTION_OP op=output_projection$' "$OUTPUT/alice.log"
grep -q '^VIT_ATTENTION_NUMERICAL_MATCH status=PASS$' "$OUTPUT/alice.log"
grep -q '^PLAINTEXT_INTERMEDIATE_RECONSTRUCTIONS=0$' "$OUTPUT/alice.log"
grep -q '^SCALAR_FALLBACKS=0$' "$OUTPUT/alice.log"
grep -q '^GENERIC_OT_MATMUL_CALLS=0$' "$OUTPUT/alice.log"
grep -q '^M2_VIT_ATTENTION_PROTOCOL=PASS$' "$OUTPUT/alice.log"
cat "$OUTPUT/alice.log"
