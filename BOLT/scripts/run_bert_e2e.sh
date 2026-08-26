#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${BINARY:-$ROOT/SCI/build_vit/bin/BOLT_BERT}"
BERT_DATA="${BERT_DATA:-${1:-}}"
OUTPUT="${OUTPUT:-$ROOT/outputs/bert_e2e}"
# BOLT_BERT opens PORT plus 64 nonlinear channels.  Keep the default range
# below Linux's usual ephemeral client-port range (32768+) for reliable
# same-host two-party runs.
PORT="${PORT:-18000}"
SAMPLE_ID="${SAMPLE_ID:-0}"
NUM_SAMPLE="${NUM_SAMPLE:-1}"
PRUNE="${PRUNE:-0}"

if [[ -z "$BERT_DATA" ]]; then
  echo "usage: BERT_DATA=/path/to/data $0" >&2
  echo "expected files under /path/to/data/weights_txt/" >&2
  exit 2
fi
if [[ ! -x "$BINARY" ]]; then
  echo "missing BOLT_BERT binary: $BINARY" >&2
  echo "build it with: TARGET=BOLT_BERT scripts/build.sh" >&2
  exit 2
fi

python3 "$ROOT/scripts/preflight_bert_data.py" \
  --data "$BERT_DATA" --role both --id "$SAMPLE_ID" \
  --num-sample "$NUM_SAMPLE"

mkdir -p "$OUTPUT"
alice_pid=""
cleanup() {
  if [[ -n "$alice_pid" ]] && kill -0 "$alice_pid" 2>/dev/null; then
    kill "$alice_pid" 2>/dev/null || true
    wait "$alice_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$BINARY" r=1 p="$PORT" path="$BERT_DATA" id="$SAMPLE_ID" \
  num_sample="$NUM_SAMPLE" prune="$PRUNE" \
  >"$OUTPUT/alice.log" 2>&1 &
alice_pid=$!

"$BINARY" r=2 p="$PORT" ip=127.0.0.1 path="$BERT_DATA" \
  id="$SAMPLE_ID" num_sample="$NUM_SAMPLE" prune="$PRUNE" \
  output="$OUTPUT/results.csv" >"$OUTPUT/bob.log" 2>&1
wait "$alice_pid"
alice_pid=""

grep -q '^BOLT_BERT_PREFLIGHT=PASS role=ALICE' "$OUTPUT/alice.log"
grep -q '^BOLT_BERT_PREFLIGHT=PASS role=BOB' "$OUTPUT/bob.log"
test -s "$OUTPUT/results.csv"

echo "BOLT_BERT_E2E=PASS"
echo "results=$OUTPUT/results.csv"
