#!/usr/bin/env bash
set -euo pipefail

# IRON GPU keeps IRON's compact BFV+MPC graph; only the ALICE HE evaluator
# runs on Phantom/CUDA.  The image remains BOB-owned, as in the CPU runner.
IRON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$IRON_ROOT/.." && pwd)"
BLB_ROOT="${BLB_ROOT:-$WORKSPACE_ROOT/BLB}"
SHARED_DATASET="${SHARED_DATASET:-$WORKSPACE_ROOT/datasets/vit_b16_shared}"

if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  echo "IRON GPU experiments are restricted to physical GPU 0; unset CUDA_VISIBLE_DEVICES or set it to 0" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=0

export BINARY="${BINARY:-$IRON_ROOT/build/secure_vit/bin/IRON_VIT_GPU}"
export INSPECTOR="${INSPECTOR:-$IRON_ROOT/build/secure_vit/bin/IRON_VIT_GPU_MODEL_INSPECT}"
export BUILD_ROOT="${BUILD_ROOT:-$WORKSPACE_ROOT/BLB/build_vit_b16_gpu}"

MODEL="${MODEL:-$WORKSPACE_ROOT/BOLT/checkpoints/vit_b16/bolt_model.txt}"
IMAGE="${IMAGE:-$SHARED_DATASET/images/image_000.png}"
MANIFEST="${MANIFEST:-$WORKSPACE_ROOT/BOLT/checkpoints/vit_b16/manifest.json}"
OUTPUT="${OUTPUT:-$IRON_ROOT/outputs/vit_b16_gpu}"
PORT="${PORT:-49020}"
PYTHON="${PYTHON:-$WORKSPACE_ROOT/BOLT/.venv-vit-b16/bin/python}"

MODEL="$MODEL" IMAGE="$IMAGE" MANIFEST="$MANIFEST" OUTPUT="$OUTPUT" \
  PORT="$PORT" PYTHON="$PYTHON" BINARY="$BINARY" INSPECTOR="$INSPECTOR" \
  BUILD_ROOT="$BUILD_ROOT" \
  bash "$BLB_ROOT/repro/run_vit_b16.sh"
