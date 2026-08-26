#!/usr/bin/env bash
set -euo pipefail

# One-command entry point after the prepared B16 assets are present.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT/.." && pwd)"
SHARED_DATASET="${SHARED_DATASET:-$WORKSPACE_ROOT/datasets/vit_b16_shared}"
IMAGE="${1:-${IMAGE:-$SHARED_DATASET/images/image_000.png}}"
OUTPUT="${2:-$ROOT/outputs/vit_b16_e2e}"
PORT="${3:-49000}"
MODEL="$ROOT/checkpoints/vit_b16/bolt_model.txt"
MANIFEST="$ROOT/checkpoints/vit_b16/manifest.json"

if [[ ! -s "$MODEL" ]]; then
  echo "BOLT ViT checkpoint not found or empty: $MODEL" >&2
  exit 1
fi
if [[ ! -s "$MANIFEST" ]]; then
  echo "BOLT ViT manifest not found or empty: $MANIFEST" >&2
  exit 1
fi
if [[ ! -f "$IMAGE" ]]; then
  echo "BOLT ViT input image not found: $IMAGE" >&2
  exit 1
fi

echo "BOLT ViT prepared runner"
echo "  image:  $IMAGE"
echo "  output: $OUTPUT"
echo "  port:   $PORT"

MODEL="$MODEL" IMAGE="$IMAGE" MANIFEST="$MANIFEST" \
  OUTPUT="$OUTPUT" PORT="$PORT" \
  "$ROOT/scripts/run_vit_b16_e2e.sh"
