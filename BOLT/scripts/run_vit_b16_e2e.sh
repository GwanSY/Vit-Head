#!/usr/bin/env bash
set -euo pipefail

# Formal two-party ViT runner.  MODEL is ALICE-only; IMAGE is BOB-only.  The
# wrapper never passes the model file to the BOB process and never enables the
# forbidden full-logit debug option.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT/.." && pwd)"
BINARY="${BINARY:-$ROOT/SCI/build_vit/bin/BOLT_VIT}"
SHARED_DATASET="${SHARED_DATASET:-$WORKSPACE_ROOT/datasets/vit_b16_shared}"
IMAGE="${IMAGE:-$SHARED_DATASET/images/image_000.png}"
MODEL="${MODEL:-$ROOT/checkpoints/vit_b16/bolt_model.txt}"
MANIFEST="${MANIFEST:-$ROOT/checkpoints/vit_b16/manifest.json}"
OUTPUT="${OUTPUT:-$ROOT/outputs/vit_b16_e2e}"
PORT="${PORT:-49000}"
PYTHON="${PYTHON:-$ROOT/.venv-vit-b16/bin/python}"

log() {
  printf '[BOLT ViT %s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

[[ -x "$BINARY" ]] || fail "executable not found: $BINARY"
[[ -s "$MODEL" ]] || fail "checkpoint not found or empty: $MODEL"
[[ -f "$IMAGE" ]] || fail "input image not found: $IMAGE"
[[ -s "$MANIFEST" ]] || fail "manifest not found or empty: $MANIFEST"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || true)"
fi
[[ -n "$PYTHON" && -x "$PYTHON" ]] || fail "Python interpreter not found"
command -v ss >/dev/null 2>&1 || fail "the 'ss' command is required to observe the ALICE listener"
if ss -H -ltn "sport = :$PORT" 2>/dev/null | grep -q .; then
  fail "TCP port $PORT is already in use; pass another port as the third argument"
fi
mkdir -p "$OUTPUT"

log "preprocessing BOB image: $IMAGE"
"$PYTHON" "$ROOT/scripts/preprocess_vit_image.py" "$IMAGE" \
  --manifest "$MANIFEST" --output "$OUTPUT/bob_image.txt"
log "preprocessing complete: $OUTPUT/bob_image.txt"

alice_pid=""
heartbeat_pid=""
cleanup() {
  if [[ -n "$heartbeat_pid" ]] && kill -0 "$heartbeat_pid" 2>/dev/null; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
  fi
  if [[ -n "$alice_pid" ]] && kill -0 "$alice_pid" 2>/dev/null; then
    kill "$alice_pid" 2>/dev/null || true
    wait "$alice_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

model_bytes="$(stat -c '%s' "$MODEL")"
log "starting ALICE; loading ${model_bytes}-byte checkpoint before listening"
log "checkpoint parsing can take about a minute; progress will appear below"
"$BINARY" r=1 p="$PORT" model="$MODEL" \
  output="$OUTPUT/prediction.txt" trace_dir="$OUTPUT" debug_logits=0 \
  > >(tee "$OUTPUT/alice.log") 2>&1 &
alice_pid=$!

listener_ready=0
for attempt in $(seq 1 600); do
  if ! kill -0 "$alice_pid" 2>/dev/null; then
    alice_status=0
    wait "$alice_pid" || alice_status=$?
    alice_pid=""
    fail "ALICE exited before listening (status $alice_status); inspect $OUTPUT/alice.log"
  fi
  if ss -H -ltn "sport = :$PORT" 2>/dev/null | grep -q .; then
    listener_ready=1
    break
  fi
  if (( attempt % 10 == 0 )); then
    log "ALICE is still loading the checkpoint (${attempt}s elapsed)"
  fi
  sleep 1
done
[[ "$listener_ready" -eq 1 ]] || fail "ALICE did not listen on port $PORT within 600 seconds"
log "ALICE is listening on 127.0.0.1:$PORT; starting BOB"

run_started_epoch="$(date +%s)"
(
  while kill -0 "$alice_pid" 2>/dev/null; do
    sleep 30
    if kill -0 "$alice_pid" 2>/dev/null; then
      now_epoch="$(date +%s)"
      log "secure inference is running ($((now_epoch - run_started_epoch))s elapsed); detailed output: $OUTPUT"
    fi
  done
) &
heartbeat_pid=$!

"$BINARY" r=2 p="$PORT" ip=127.0.0.1 input="$OUTPUT/bob_image.txt" \
  output="$OUTPUT/prediction.txt" trace_dir="$OUTPUT" debug_logits=0 \
  2>&1 | tee "$OUTPUT/bob.log"
wait "$alice_pid"
alice_pid=""
if [[ -n "$heartbeat_pid" ]] && kill -0 "$heartbeat_pid" 2>/dev/null; then
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
fi
heartbeat_pid=""

log "secure execution finished; comparing against the fixed-point clear reference"
reference_args=(
  --model "$MODEL"
  --input "$OUTPUT/bob_image.txt"
  --secure-prediction "$OUTPUT/prediction.txt"
  --output "$OUTPUT/reference_report.json"
)
# The protocol supports a validation-only layer prefix.  The clear reference
# must use the same prefix; otherwise a one-layer secure run is incorrectly
# compared with the full 12-layer model.
if [[ -n "${BOLT_VIT_MAX_LAYERS:-}" ]]; then
  reference_args+=(--max-layers "$BOLT_VIT_MAX_LAYERS")
fi
"$PYTHON" "$ROOT/scripts/vit_reference.py" "${reference_args[@]}" \
  | tee "$OUTPUT/reference.log"

grep -q '^PLAINTEXT_INTERMEDIATE_RECONSTRUCTIONS=0$' "$OUTPUT/alice.log"
grep -q '^SCALAR_FALLBACKS=0$' "$OUTPUT/alice.log"
grep -q '^GENERIC_OT_MATMUL_CALLS=0$' "$OUTPUT/alice.log"
grep -q '^SILENT_FALLBACKS=0$' "$OUTPUT/alice.log"
test -s "$OUTPUT/backend_breakdown.csv"
grep -q '^he,' "$OUTPUT/backend_breakdown.csv"
grep -q '^mpc,' "$OUTPUT/backend_breakdown.csv"
grep -q '^conversion,' "$OUTPUT/backend_breakdown.csv"
log "all fail-closed protocol checks completed"
echo "BOLT_VIT_B16_E2E=COMPLETED_WITH_LAYERWISE_SECURE_METRICS_UNAVAILABLE"
