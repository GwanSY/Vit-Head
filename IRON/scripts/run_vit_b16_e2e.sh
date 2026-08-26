#!/usr/bin/env bash
set -euo pipefail

# IRON's standard ViT-B/16 entry point.  The execution graph is the shared
# packed BOLT adapter; this wrapper only selects the IRON_VIT executable and
# keeps the model/input ownership split explicit.
IRON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$IRON_ROOT/.." && pwd)"
BLB_ROOT="${BLB_ROOT:-$WORKSPACE_ROOT/BLB}"
SHARED_DATASET="${SHARED_DATASET:-$WORKSPACE_ROOT/datasets/vit_b16_shared}"
if [[ "${VALIDATION_LOGITS:-0}" == "1" || "${AGGREGATE_ERRORS:-0}" == "1" ]]; then
  default_binary="$IRON_ROOT/build/secure_vit/bin/IRON_VIT_VALIDATE"
else
  default_binary="$IRON_ROOT/build/secure_vit/bin/IRON_VIT"
fi
BINARY="${BINARY:-$default_binary}"
MODEL="${MODEL:-$WORKSPACE_ROOT/BOLT/checkpoints/vit_b16/bolt_model.txt}"
IMAGE="${IMAGE:-$SHARED_DATASET/images/image_000.png}"
MANIFEST="${MANIFEST:-$WORKSPACE_ROOT/BOLT/checkpoints/vit_b16/manifest.json}"
OUTPUT="${OUTPUT:-$IRON_ROOT/outputs/vit_b16_e2e}"
PORT="${PORT:-49010}"
INSPECTOR="${INSPECTOR:-$IRON_ROOT/build/secure_vit/bin/IRON_VIT_MODEL_INSPECT}"
PYTHON="${PYTHON:-$WORKSPACE_ROOT/BOLT/.venv-vit-b16/bin/python}"

[[ -x "$BINARY" ]] || { echo "missing IRON_VIT: $BINARY" >&2; exit 1; }
[[ -x "$INSPECTOR" ]] || { echo "missing IRON_VIT inspector: $INSPECTOR" >&2; exit 1; }
[[ -s "$MODEL" ]] || { echo "missing ViT model: $MODEL" >&2; exit 1; }
[[ -s "$MANIFEST" ]] || { echo "missing checkpoint manifest: $MANIFEST" >&2; exit 1; }
[[ -f "$IMAGE" ]] || { echo "missing image: $IMAGE" >&2; exit 1; }

BINARY="$BINARY" INSPECTOR="$INSPECTOR" BUILD_ROOT="$BLB_ROOT/build_vit_b16" \
  MODEL="$MODEL" IMAGE="$IMAGE" MANIFEST="$MANIFEST" \
  OUTPUT="$OUTPUT" PORT="$PORT" PYTHON="$PYTHON" \
  bash "$BLB_ROOT/repro/run_vit_b16.sh"
