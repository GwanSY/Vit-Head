#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${BINARY:-$ROOT/SCI/build_vit/bin/BERT_SHARED_BACKEND_REGRESSION}"
OUTPUT="${OUTPUT:-$ROOT/outputs/regression}"
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

"$BINARY" r=1 p="$PORT" output="$OUTPUT/bert_shared_backend.csv" \
  >"$OUTPUT/bert_shared_backend_alice.log" 2>&1 &
alice_pid=$!
"$BINARY" r=2 p="$PORT" ip=127.0.0.1 \
  output="$OUTPUT/bert_shared_backend.csv" \
  >"$OUTPUT/bert_shared_backend_bob.log" 2>&1
wait "$alice_pid"
alice_pid=""

grep -q '^BERT_SHARED_BACKEND=PASS$' \
  "$OUTPUT/bert_shared_backend_alice.log"
grep -q ',PASS$' "$OUTPUT/bert_shared_backend.csv"
cat "$OUTPUT/bert_shared_backend.csv"
cat "$OUTPUT/bert_shared_backend_alice.log"
