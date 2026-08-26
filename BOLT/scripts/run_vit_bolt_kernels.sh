#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${BINARY:-$ROOT/SCI/build_vit/bin/BOLT_VIT_KERNEL_TEST}"
OUTPUT="${OUTPUT:-$ROOT/outputs/vit_bolt_kernels}"
PORT="${PORT:-47200}"

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

"$BINARY" r=1 p="$PORT" >"$OUTPUT/alice.log" 2>&1 &
alice_pid=$!
"$BINARY" r=2 p="$PORT" ip=127.0.0.1 >"$OUTPUT/bob.log" 2>&1
wait "$alice_pid"
alice_pid=""

grep -q '^BOLT_BACKEND_TEST op=ct_pt_matmul$' "$OUTPUT/alice.log"
grep -q '^BOLT_BACKEND_TEST op=qk_ct_ct$' "$OUTPUT/alice.log"
grep -q '^BOLT_BACKEND_TEST op=softmax_v_ct_ct$' "$OUTPUT/alice.log"
test "$(grep -c '^status=PASS$' "$OUTPUT/alice.log")" -eq 3
grep -q '^VIT_BOLT_KERNELS=PASS$' "$OUTPUT/alice.log"
cat "$OUTPUT/alice.log"
